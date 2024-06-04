import random
from typing import List, Tuple
from collections import defaultdict


class Evaluator:
    def __init__(self) -> None:
        self.answer_marker = "answer is"

    def _is_number(self, s) -> Tuple[bool, str]:    
        try:      
            res = float(s)        
            return True, str(res)
        except:  
            pass  
        try:        
            import unicodedata  
            res = unicodedata.numeric(s) 
            return True, str(res)
        except:        
            pass    
        return False, None 

    def validate_completion(self, completion: str) -> bool:
        if self.answer_marker.lower() in completion.lower():
            return True
        return False
    
    def isolate_answer(self, text: str):
        if text is None:
            return None
        
        assert isinstance(text, str)
        text = text.lower()
        split_ans = text.split(self.answer_marker.lower())
        if len(split_ans) > 1:
            ans = split_ans[-1].replace(":", "").strip()
            extract_ans_temp = ans.split(".\n")[0].strip()
            if len(extract_ans_temp) > 0 and extract_ans_temp[-1] == ".":
                extract_ans = extract_ans_temp[0:-1]
            else:
                extract_ans = extract_ans_temp
            extract_ans = extract_ans.strip().strip("\n")
            return extract_ans
        else:
            return None
    
    def find_most_confident_answer(self, completions: List[str], prior_weights: List[float] = None):
        """Returns the most confident answer, its completion, its id in the input list, and its confidence."""
        if completions is None or len(completions) == 0:
            return None, None, None, None
        
        answer2completions = defaultdict(list)
        answer2ids = defaultdict(list)
        for id, c in enumerate(completions):
            try:
                model_answer = self.extract_answer_from_model_completion(c)
                has_existed = False
                for existing_answer in answer2completions.keys():
                    if self.check_answers_equiv(model_answer, existing_answer):
                        assert not has_existed
                        has_existed = True
                        answer2completions[existing_answer].append(c)
                        answer2ids[existing_answer].append(id)       
                if not has_existed:
                    answer2completions[model_answer].append(c)
                    answer2ids[model_answer].append(id)         
            except:
                pass

        assert len(answer2completions.keys()) > 0, "There are no valid completions."
        if prior_weights is not None:
            assert len(completions) == len(prior_weights)
            completion2count = {}
            for answer, answer_completions in answer2completions.items():
                count = len(answer_completions)
                for answer_completion in answer_completions:
                    completion2count[answer_completion] = count
            
            completion2score = {}
            for id, (completion, count) in enumerate(completion2count.items()):
                prior_weight = prior_weights[id]
                score = prior_weight * (count / len(completions))
                completion2score[completion] = score
            
            most_confident_completion = max(completion2score.keys(), key=lambda x: completion2score[x])
            
            return self.extract_answer_from_model_completion(most_confident_completion), most_confident_completion, completions.index(most_confident_completion), completion2score[most_confident_completion]
        else:
            most_confident_answer = max(answer2completions.keys(), key=lambda x: len(answer2completions[x]))
            assert len(answer2completions[most_confident_answer]) > 0, "There are no completions for the most confident answer."
            confidence = len(answer2completions[most_confident_answer]) / len(completions)
            assert confidence > 0
            return most_confident_answer, answer2completions[most_confident_answer][0], answer2ids[most_confident_answer][0], confidence

    def mmy_select_answer(self, completion2score, answer2completions, completions, answer_selection_mode, topk):
        answer2score = {}
        answer_counts = {}
        for completion, score in completion2score.items():
            answer = self.extract_answer_from_model_completion(completion)
            if answer in answer2score:
                answer2score[answer] += score
                answer_counts[answer] += 1
            else:
                answer2score[answer] = score
                answer_counts[answer] = 1
        
        for answer in answer2score:
            answer2score[answer] /= answer_counts[answer]
        
        if answer_selection_mode == "topk":
            top_answers = sorted(answer2score.items(), key=lambda x: x[1], reverse=True)[:topk]
            answers, scores = zip(*top_answers)
            total_score = sum(scores)
            try:
                probabilities = [score / total_score for score in scores]
                selected_answer = random.choices(answers, weights=probabilities, k=1)[0]
            except:
                selected_answer = random.choices(answers, k=1)[0]
        elif answer_selection_mode == "adaptive":
            threshold = 0.3
            top_answer = max(answer2score, key=answer2score.get)
            if answer2score[top_answer] / sum(answer2score.values()) >= threshold:
                selected_answer = top_answer
            else:
                answers = list(answer2score.keys())
                selected_answer = random.choice(answers)
        
        most_confident_completion = answer2completions[selected_answer][0]
        completion_index = completions.index(most_confident_completion)
        confidence = answer2score[selected_answer]

        return selected_answer, most_confident_completion, completion_index, confidence

    def mmy_calculate_completion_scores(self, prior_weights, answer2completions):
        completion2count = {}
        for answer, comps in answer2completions.items():
            count = len(comps)
            for comp in comps:
                completion2count[comp] = count

        completion2score = {}
        for idx, comp in enumerate(completion2count.keys()):
            weight = prior_weights[idx] if prior_weights is not None else 1
            score = weight * completion2count[comp]
            completion2score[comp] = score
        return completion2score
    
    def mmy_select_response(self, completion2score, completions, answer_selection_mode, topk):
        if answer_selection_mode == "topk":
            sorted_completions = sorted(completion2score.items(), key=lambda x: x[1], reverse=True)[:topk]
            completions, scores = zip(*sorted_completions)
            total_score = sum(scores)
            try:
                probabilities = [score / total_score for score in scores]
                sampled_completion = random.choices(completions, weights=probabilities, k=1)[0]
            except:
                sampled_completion = random.choices(completions, k=1)[0]
            confidence = completion2score[sampled_completion]
            most_confident_answer = self.extract_answer_from_model_completion(sampled_completion)
            id_of_most_confident = completions.index(sampled_completion)
            return most_confident_answer, sampled_completion, id_of_most_confident, confidence

        elif answer_selection_mode == "adaptive":
            top_score = max(completion2score.values())
            total_score = sum(completion2score.values())
            score_ratio = top_score / total_score
            
            if score_ratio >= 0.3:
                most_confident_completion = max(completion2score, key=completion2score.get)
                confidence = completion2score[most_confident_completion]
                most_confident_answer = self.extract_answer_from_model_creation(most_confident_completion)
                id_of_most_confident = list(completion2score.keys()).index(most_confident_completion)
            else:
                possible_completions = list(completion2score.keys())
                sampled_completion = random.choice(possible_completions)
                confidence = completion2score[sampled_completion]
                most_confident_answer = self.extract_answer_from_model_completion(sampled_completion)
                id_of_most_confident = possible_completions.index(sampled_completion)
            
            return most_confident_answer, sampled_completion, id_of_most_confident, confidence

        else:
            raise NotImplementedError("The selection mode provided is not implemented.")

    def mmy_find_most_confident_answer(
            self, 
            completions: List[str], 
            answer_selection_metric: str,
            answer_selection_mode: str,
            topk,
            prior_weights: List[float] = None
        ):

        if not completions or len(completions) == 0:
            return None, None, None, None

        answer2completions = defaultdict(list)
        for idx, comp in enumerate(completions):
            try:
                answer = self.extract_answer_from_model_completion(comp)
                answer2completions[answer].append(comp)
            except:
                continue

        if not answer2completions:
            return None, None, None, None

        completion2score = self.mmy_calculate_completion_scores(prior_weights, answer2completions)

        if answer_selection_metric == "select_answer":
            most_confident_answer, sampled_completion, id_of_most_confident, confidence = self.mmy_select_answer(completion2score, answer2completions, completions, answer_selection_mode, topk)
            return most_confident_answer, sampled_completion, id_of_most_confident, confidence
        elif answer_selection_metric == "select_response":
            most_confident_answer, sampled_completion, id_of_most_confident, confidence = self.mmy_select_response(completion2score, completions, answer_selection_mode, topk)
            return most_confident_answer, sampled_completion, id_of_most_confident, confidence
        else:
            raise NotImplementedError

    def check_answers_equiv(self, answer_a: str, answer_b: str):
        raise NotImplementedError
    
    def extract_answer_from_gold_solution(self, solution: str) -> str:
        raise NotImplementedError
    
    def extract_answer_from_model_completion(self, completion: str) -> str:
        raise NotImplementedError
    
    def judge_answer(self, completion: str, gold_solution: str) -> bool:
        is_number, _ = self._is_number(completion)
        if is_number:
            return completion, self.check_answers_equiv(completion, gold_solution)
        pred = self.extract_answer_from_model_completion(completion.strip().split("\n")[-1])
        return pred, self.check_answers_equiv(pred, gold_solution)


# if __name__ == "__main__":
#     folio_evaluator = MMLUSTEMEvaluator()
    
#     model_completion = "Let's think step by step. We have \[ det(A^2) = (det(A))^2 \geq 0,\] hence II holds. III is false: as a counterexample take a diagonal matrix with -1 and 1 on the diagonal. Then $A^2$ is the identity matrix. The answer is B."
#     model_answer = folio_evaluator.extract_answer_from_model_completion(model_completion)
    
#     gold_solution = "B"
#     gold_answer = folio_evaluator.extract_answer_from_gold_solution(gold_solution)
    
#     correctness = folio_evaluator.check_answers_equiv(model_answer, gold_answer)
    
#     print(correctness)  # True
