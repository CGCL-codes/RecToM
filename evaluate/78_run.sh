mkdir -p ../logs/llama3.1-8B
timestamp=$(date +%Y%m%d_%H%M)

logfile="../logs/llama3.1-8B/8_belief_rec_${timestamp}.log"
echo "Log file:$logfile"
nohup python 78_ds_RecommenderToM.py --dataset_type  8_belief_rec_2_com.json  --model 'meta-llama/llama-3.1-8b-instruct' 2>&1 | tee -a $logfile &


