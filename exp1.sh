## Need to Manually Switch the Baseline to repeat experiments
# Train Bayesian, Imitation baselines in Isolation
python main.py --case casino --stadium config_bayesian --leng_limit 20 --train_rounds 5000 --test_rounds 500 --vb_highest 10 --vb_lowest 10 --vb_collect --save;
python main.py --case casino --stadium config_imitation --leng_limit 20 --train_rounds 5000 --test_rounds 500 --vb_highest 10 --vb_lowest 10 --vb_collect --save;

# Train MCTS in Isolation
# Train Q Learning in Isolation
# Train Deep Q Learning in Isolation
python main.py --case casino --stadium config_mcts --leng_limit 20 --train_rounds 5000 --test_rounds 500 --vb_highest 10 --vb_lowest 10 --vb_collect --save;
python main.py --case casino --stadium config_qlearning --leng_limit 20 --train_rounds 5000 --test_rounds 500 --vb_highest 10 --vb_lowest 10 --vb_collect --save;
python main.py --case casino --stadium config_deepqlearning --leng_limit 20 --train_rounds 5000 --test_rounds 500 --vb_highest 10 --vb_lowest 10 --vb_collect --save;

# Re-train Bayesian and Imitation
python main.py --case casino --stadium config_bayesian --leng_limit 20 --train_rounds 5000 --test_rounds 500 --vb_highest 10 --vb_lowest 10 --vb_collect --save;
python main.py --case casino --stadium config_imitation --leng_limit 20 --train_rounds 5000 --test_rounds 500 --vb_highest 10 --vb_lowest 10 --vb_collect --save;

# Test MCTS vs Baseline
# Test Q Learning vs Baseline
# Test Deep Q Learning vs Baseline
python main.py --case casino --stadium config_mcts --leng_limit 20 --train_rounds 0 --test_rounds 500 --vb_highest 10 --vb_lowest 10 --vb_collect --load;
python main.py --case casino --stadium config_qlearning --leng_limit 20 --train_rounds 0 --test_rounds 500 --vb_highest 10 --vb_lowest 10 --vb_collect --load;
python main.py --case casino --stadium config_deepqlearning --leng_limit 20 --train_rounds 0 --test_rounds 500 --vb_highest 10 --vb_lowest 10 --vb_collect --load;

# Test all Isolation Trained Models in Society
python main.py --case casino --stadium config_all_isolation_test --leng_limit 20 --train_rounds 0 --test_rounds 500 --vb_highest 10 --vb_lowest 10 --vb_collect;

# Train all Agents in Society, Save Only One
# Train all Agents in Society, Save Only Two
# Train all Agents in Society, Save Only Three
python main.py --case casino --stadium config_all_society_train --leng_limit 20 --train_rounds 5000 --test_rounds 500 --vb_highest 10 --vb_lowest 10 --vb_collect --save --save_selected 1;
python main.py --case casino --stadium config_all_society_train --leng_limit 20 --train_rounds 5000 --test_rounds 500 --vb_highest 10 --vb_lowest 10 --vb_collect --save --save_selected 2;
python main.py --case casino --stadium config_all_society_train --leng_limit 20 --train_rounds 5000 --test_rounds 500 --vb_highest 10 --vb_lowest 10 --vb_collect --save --save_selected 3;

# Test MCTS Society vs Baseline
# Test Q Learning Society vs Baseline
# Test Deep Q Learning Society vs Baseline
python main.py --case casino --stadium config_mctssociety_test --leng_limit 20 --train_rounds 0 --test_rounds 500 --vb_highest 10 --vb_lowest 10 --vb_collect;
python main.py --case casino --stadium config_qlearningsociety_test --leng_limit 20 --train_rounds 0 --test_rounds 500 --vb_highest 10 --vb_lowest 10 --vb_collect;
python main.py --case casino --stadium config_deepqlearningsociety_test --leng_limit 20 --train_rounds 0 --test_rounds 500 --vb_highest 10 --vb_lowest 10 --vb_collect;

# Re-train Bayesian and Imitation
python main.py --case casino --stadium config_bayesian --leng_limit 20 --train_rounds 5000 --test_rounds 500 --vb_highest 10 --vb_lowest 10 --vb_collect --save;
python main.py --case casino --stadium config_imitation --leng_limit 20 --train_rounds 5000 --test_rounds 500 --vb_highest 10 --vb_lowest 10 --vb_collect --save;

# Test MCTS Society vs Baseline
# Test Q Learning Society vs Baseline
# Test Deep Q Learning Society vs Baseline
python main.py --case casino --stadium config_mctssociety_test --leng_limit 20 --train_rounds 0 --test_rounds 500 --vb_highest 10 --vb_lowest 10 --vb_collect;
python main.py --case casino --stadium config_qlearningsociety_test --leng_limit 20 --train_rounds 0 --test_rounds 500 --vb_highest 10 --vb_lowest 10 --vb_collect;
python main.py --case casino --stadium config_deepqlearningsociety_test --leng_limit 20 --train_rounds 0 --test_rounds 500 --vb_highest 10 --vb_lowest 10 --vb_collect;

# Test all Society Trained Models in Society
python main.py --case casino --stadium config_all_society_train --leng_limit 20 --train_rounds 0 --test_rounds 500 --vb_highest 10 --vb_lowest 10 --vb_collect --load;
