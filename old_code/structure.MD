
For each update time we have a new policy:

1. policy2esteqn: dictionary that maps the policy update number to data to form the estimating equation for the estimators used by that policy. The data that will be used to form these estimating equations can include:
    - estimators used by the policy
    - reward
    - action
    - action selection probabilities
    - states
    - availability
    - calender decision time
    - user decision time
    - design matrix
    - user ids

2. policy2collect: dictionary that maps the policy update number to the data that was collected using that policy. The data that will be used to form the action selection probabilities can include:
    - estimators used by the policy
    - reward
    - action
    - action selection probabilities
    - states
    - availability
    - calender decision time
    - user decision time
    - design matrix
    - user ids
    - randomness used by the algorithm

3. get_est_eqn: takes teh estimators used by the policy and the policy2esteqn data for that policy and outputs the empirical estimating equations

4. get_action_probs_inner: takes estimators used by the policy and the policy2collect data for that policy and outputs action selection probabilities


