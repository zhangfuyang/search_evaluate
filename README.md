# search_evaluate

## [Feb 8]

### Changes (Xu):
> agent.py:
>> Transition
>
>> ReplayMemory
>
>> Agent
>
>>> selection_action()


## [Feb 7]

### Changes (Xu):

- modified new_search_with_train.py to using policy_net and target_net
- lock is changed to a global counter
- remove adding ground-truth edge action from candidate_enumerate_training() in new_utils.py

### Changes (Zhang):

- fix a bug in new_dataset.py, where finding the same edge in next_edges part

- replace random search policy as beam search in searchThread, with 20% percent to random explore

- put test() and train() into a new file new_train_test_func.py

- add TD version of crossloss in train()

- add some comment in new_config.py

> new_config.py:
>> use_heatmap=False. -> no heatmap prediction as well as crossloss
>
>> use_heatmap=True, use_cross_loss=False. -> activate heatmap prediction, but no crossloss
> 
>> use_heatmap=True, use_cross_loss=True. -> activate heatmap prediction as well as crossloss
## [Feb 6]

### Changes (Xu):

- modified get_score() in new_scoreAgent.py, current graph score is computed based on value function (not rewards).

- changed corner and edge evaluator models from classification to value regression.

- changed cornerloss and edgeloss to SmoothL1Loss in new_earch_with_train.py.

- added discounted rewards for corner and edge loss in new_earch_with_train.py.



### Changes (Zhang):

- make a get_score_list() function in new_score_Agent.py, which input is a list of candidates. This function is for reusing same img_volume and heatmap, no need to compute for each candidate (they are same), so save a bit of time.

- write test() function for testing. We randomly pick 10 graphs for searching (width=2, depth=6), and return f1 score of edge

- split two files new_search_with_train.py (1 search thread), new_search_with_train_multi_thread.py (2 search threads)



