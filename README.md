# search_evaluate

[Feb 6]

Changes (Xu):

(1) modified get_score() in new_scoreAgent.py, current graph score is computed based on value function (not rewards).

(2) changed corner and edge evaluator models from classification to value regression.

(3) changed cornerloss and edgeloss to SmoothL1Loss in new_earch_with_train.py.

(4) added discounted rewards for corner and edge loss in new_earch_with_train.py.
