import io
import os
import random
import re
import warnings
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm, trange

from .mcts import MCTS, MCTSNode
from .models import QueryLM


def is_terminal_question(prompt, prompt_index, overall_question=None):
    prompt = prompt.split('\n\n')[-1]
    if 'Now we can answer' in prompt:
        return True
    question = prompt.split('\n')[0]
    if f'Question {prompt_index}.' not in prompt:
        return False
    last_sub = prompt.split(f'Question {prompt_index}.')[-1].split('\n')[0]
    # if (overall_question is not None and overall_question.lower() in last_sub.lower()) or last_sub.lower() in question.lower():
    if last_sub.lower() in question.lower():
        return True
    return False


class ReasoningMCTSNode(MCTSNode):
    @property
    def visited(self):
        return self._visited

    def __init__(self, prompt, question_prompt, gen_fn, reward_fn, depth, r1_default, r_alpha, prompt_index,
                 parent: 'ReasoningMCTSNode' = None, r0=0., overall_question=None, random_temp_log=None, extract_answer_fn=None):
        self._conf = None
        self.children = []
        self.prompt = prompt
        self.question_prompt = question_prompt
        self.gen_fn = gen_fn
        self.reward_fn = reward_fn
        self.depth = depth
        self._r0 = r0
        self._r1 = self._r1_default = r1_default
        self._r_alpha = r_alpha
        self._ans_list = None
        self._visited = False
        self.parent = parent
        self._prompt_index = prompt_index
        self.overall_question = overall_question
        self.random_temp_log = random_temp_log
        self.extract_answer_fn = extract_answer_fn

    def _child_node(self, prompt, question_prompt, r0):
        child = ReasoningMCTSNode(prompt, question_prompt, self.gen_fn, self.reward_fn, self.depth + 1,
                                 self._r1_default, self._r_alpha, self._prompt_index, parent=self, r0=r0,
                                 overall_question=self.overall_question, random_temp_log=self.random_temp_log,
                                 extract_answer_fn=self.extract_answer_fn)

        # Output all terminal answer in MCTS rollouts
        if self.random_temp_log is not None and is_terminal_question(child.prompt, child._prompt_index, child.overall_question):
            warnings.warn('Ignore following warning ....')

            node_temp = deepcopy(child)
            node_temp.reward_fn = deepcopy(child.reward_fn)
            ans_temp, _, _ = node_temp.reward_fn(node_temp.prompt, node_temp.depth)

            ans_temp = ans_temp.split('\n\n')[-1].strip().split("\n")[-1]
            # import pdb; pdb.set_trace()
            t = self.extract_answer_fn(ans_temp)
            with open(self.random_temp_log, 'a') as f:
                if t is not None:
                    f.write(t + "\n")
                else:
                    f.write("None\n")
            warnings.warn('Ignore above warning ...')

        return child

    def _get_children(self):
        self._visited = True
        self._calculate_reward()
        if self.is_terminal:
            return self.children
        questions, question_prompts, r0 = self.gen_fn(self.prompt, self.question_prompt, self.depth)
        for question, qp, r in zip(questions, question_prompts, r0):
            self.children.append(self._child_node(question, qp, r))
        return self.children

    def find_children(self):
        self.children = self.children or self._get_children()
        return self.children

    def find_one_child(self) -> MCTSNode:
        return random.choice(self.find_children())

    def _calculate_reward(self):
        self.prompt, self._r1, self._ans_list = self.reward_fn(self.prompt, self.depth)

    def _static_terminal(self):
        return is_terminal_question(self.prompt, self._prompt_index, self.overall_question)

    @property
    def is_terminal(self):
        return self._static_terminal() or self.reward < -1

    @property
    def reward(self):
        if self._r0 < 0 or self._r1 < 0:
            return min(self._r0, self._r1)
        return self._r0 ** self._r_alpha * self._r1 ** (1 - self._r_alpha)

    def print(self, mcts: MCTS, file=None):
        def pprint(*args):
            if file is None:
                tqdm.write(*args)
            else:
                print(*args, file=file)
        p1 = '-' * (4 * self.depth - 4)
        prefix = ' ' * (4 * self.depth - 4)
        question = 'Q' + self.prompt.split(f'Question {self._prompt_index}')[-1].split('\n')[0]
        pprint(p1 + question)
        pprint(prefix + f'R: {self.reward:.3f} ; N: {mcts.N[self]} ; M: {mcts.M[self]:.3f} ; r0 : {self._r0:.3f}')
        if not self.visited:
            return
        answer = 'A' + self.prompt.split(f'Answer {self._prompt_index}')[-1].split('\n')[0]
        if self.reward < -1:
            if file is not None:
                pprint(prefix + question)
                pprint(prefix + answer)
            return
        if self.parent is not None:
            if file is not None:
                pprint(prefix + answer)
            # import pdb; pdb.set_trace()
            match = self.extract_answer_fn(answer)
            if match is not None:
                term = '\u25A1' if self.is_terminal else ''
                pprint(prefix + f'answer: {match} ; ans_list: {self._ans_list} ; r1 : {self._r1:.3f}{term}')
            # if not self.is_terminal:
            #     pprint(prefix + f'conf_list : {self._conf_list} ; r2 : {self._r2:.3f}')
        for child in self.children:
            child.print(mcts, file)
        if self.depth == 1:
            pprint("=" * 12)

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.gen_fn is None or self.reward_fn is None:
            warnings.warn('MCTSNode loaded from pickle is read-only; Do not further roll out the tree!')

    def __getstate__(self):
        state = self.__dict__.copy()
        state['gen_fn'] = None
        state['reward_fn'] = None
        return state


def reasoning_mcts_search(question: str,
                          prompts,
                          question_prompts,
                          world_model: QueryLM,
                          n_sample_subquestion,
                          temperature,
                          mcts_rollouts,
                          w_exp,
                          n_sample_confidence,
                          max_depth,
                          r_alpha,
                          r1_default,
                          eos_token_id,
                          speedup_confidence_batch_size=None,
                          task=None,
                          extract_answer_fn=None,
                          output_ans_list=False,
                          random_temp_log=None,):
    if output_ans_list: assert random_temp_log is not None
    if not output_ans_list: random_temp_log = None

    if speedup_confidence_batch_size is None:
        speedup_confidence_batch_size = n_sample_confidence
    if task in ["gsm8k", "math", "svamp"]:
        match = re.match('.*((Calculate|calculate|how|How|what|What|Find|find|True or false).*)$', question)
        overall_question = match[1] if match else question
    elif task == "folio":
        overall_question = question.split("Based on the above information, is the following statement true, false, or uncertain?")[-1].strip()
    elif task == "logiqa":
        try:
            t = question.split(" (A) ")[0].split(".")
            overall_question = t[-1] if t[-1] != "" else t[-2]
            if not overall_question.strip().startswith("Which"):
                match = re.match('.*((Which|which).*)$', overall_question)
                overall_question = match[1] if match else overall_question
            overall_question = overall_question.strip()
        except:
            overall_question = question
    elif task == "sat":
        match = re.match('.*((Calculate|calculate|how|How|what|What|Find|find|True or false).*)$', question.split(" A.")[0])
        overall_question = match[1] if match else question.split(" A. ")[0]
    elif task == "bgqa":
        overall_question = question.split("Based on the game state and the rules and preferences, ")[-1].strip()
    overall_question = overall_question[0].upper() + overall_question[1:]
    prompt_index = prompts['index']

    def gen_fn(inp, q_inp, depth):
        subquestion_prefix = prompts["subquestion_prefix"].format(depth)
        agent_input = inp + subquestion_prefix
        overall_question_output = inp + prompts["overall_question_prefix"].format(depth, overall_question)

        if depth == max_depth:
            agent_output = [overall_question_output]
        else:
            agent_output = world_model.query_LM(agent_input, do_sample=True, num_return_sequences=n_sample_subquestion,
                                                eos_token_id=eos_token_id, temperature=temperature)
            for i, output in enumerate(agent_output):
                if is_terminal_question(output, prompt_index, overall_question):
                    agent_output[i] = overall_question_output

        # unique the output
        # set does not guarantee order ; dict guarantees insertion order
        agent_output = list(dict.fromkeys(agent_output))
        questions = [o.split(subquestion_prefix)[-1] for o in agent_output]
        r0 = r0_fn(q_inp, questions, depth)

        question_output = [q_inp + question_prompts["subquestion_prefix"].format(depth) + q for q in questions]

        return agent_output, question_output, r0

    def r0_fn(q_inp, questions, depth):
        inp = [q_inp + question_prompts["new_subquestion_prefix"].format(depth) +
               q.replace('Now we can answer the question: ', '') +
               question_prompts["answer_prefix"] for q in questions]
        yes_no = world_model.query_next_token(inp)
        return yes_no[:, 0]

    def r1_fn(inp, depth):
        if f'Question {prompt_index}.' not in inp:
            return 0, inp, []
        answer_prefix = prompts["answer_prefix"].format(depth - 1)
        world_input = inp + answer_prefix

        answer_dict = defaultdict(lambda: [])
        answer_list = []
        sampled = 0
        while sampled < n_sample_confidence:
            world_output = world_model.query_LM(world_input, do_sample=True,
                                                num_return_sequences=speedup_confidence_batch_size,
                                                eos_token_id=eos_token_id, temperature=temperature)
            sampled += speedup_confidence_batch_size
            for output in world_output:
                result = output.strip().split('\n')[-1]
                # import pdb; pdb.set_trace()
                sub_answer = extract_answer_fn(result)
                if sub_answer is None:
                    continue
                answer_dict[sub_answer].append(output)
                answer_list.append(sub_answer)
            if len(answer_dict) == 0:
                continue
            sorted_answer_dict = sorted(answer_dict.items(), key=lambda p: len(p[1]), reverse=True)
            max_len = len(sorted_answer_dict[0][1])
            if max_len < 2:
                continue
            if len(sorted_answer_dict) < 2:
                break
            second_max_len = len(sorted_answer_dict[1][1])
            if max_len >= len(answer_dict) / 2 and max_len > second_max_len:
                break
        if len(answer_dict) == 0:
            return -10, output, []
        answer = sorted_answer_dict[0][1][0]  # [0]: maximum; [1]: list of outputs; [0]: first output in the list
        r1 = max_len / len(answer_list)
        return r1, answer, answer_list

    def reward_fn(inp, depth):
        r1, answer, ans_list = r1_fn(inp, depth)
        return answer, r1, ans_list

    input_prompts = prompts["input"] + prompts["question_prefix"] + question.strip() + "\n"
    input_question_prompts = question_prompts["input"] + question_prompts["question_prefix"] + question.strip() + "\n"

    mcts = MCTS(w_exp=w_exp, prior=True, aggr_reward='mean', aggr_child='max')
    root = ReasoningMCTSNode(input_prompts, input_question_prompts, gen_fn, reward_fn,
                             depth=1, r1_default=r1_default, r_alpha=r_alpha, prompt_index=prompt_index,
                             overall_question=overall_question, random_temp_log=random_temp_log, extract_answer_fn=extract_answer_fn)
    trajs = []
    trees = []
    for _ in (pbar := trange(mcts_rollouts, disable=bool(int(os.environ.get("LOCAL_RANK", -1))), position=0)):
        mcts.rollout(root)
        root.print(mcts)
        max_n, max_r = mcts.max_mean_terminal(root)
        trajs.append(traj := max_n.prompt.split('\n\n')[-1])
        # import pdb; pdb.set_trace()
        temp_r = extract_answer_fn(traj.strip().split("\n")[-1])
        if temp_r == None:
            temp_r = 'not found'
        pbar.set_description(f'{max_r:.3f} {temp_r}')
        tree_copy = deepcopy(root)
        tree_copy.Q = dict(mcts.Q)
        tree_copy.N = dict(mcts.N)
        tree_copy.M = dict(mcts.M)
        trees.append(tree_copy)

    ans_list = []
    if output_ans_list:
        try:
            with open(random_temp_log, 'r') as f:
                ans_list = [item.strip() for item in f.readlines()]
            os.system(f"rm {random_temp_log}")
        except:
            pass

    with io.StringIO() as f:
        root.print(mcts, file=f)
        tree = f.getvalue()
    return trajs, tree, trees, ans_list
