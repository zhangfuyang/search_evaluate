# search_evaluate

[Feb 6]

Changes (Xu):

(1) modified get_score() in new_scoreAgent.py, current graph score is computed based on value function (not rewards).

(2) changed corner and edge evaluator models from classification to value regression.

(3) changed cornerloss and edgeloss to SmoothL1Loss in new_earch_with_train.py.

(4) added discounted rewards for corner and edge loss in new_earch_with_train.py.



Changes (Zhang):

(1) make a get_score_list() function in new_score_Agent.py, which input is a list of candidates. This function is for reusing same img_volume and heatmap, no need to compute for each candidate (they are same), so save a bit of time.

(2) write test() function for testing. We randomly pick 10 graphs for searching (width=2, depth=6), and return f1 score of edge

(3) split two files new_search_with_train.py (1 search thread), new_search_with_train_multi_thread.py (2 search threads)
