mkdir -p ../logs/llama3.1-8B
timestamp=$(date +%Y%m%d_%H%M)

logfile="../logs/llama3.1-8B/2_intent_seeker_${timestamp}.log"
echo "Log file:$logfile"
nohup python 12_ds_RecommenderToM.py --dataset_type 2_intent_seeker.json --model 'meta-llama/llama-3.1-8b-instruct' 2>&1 | tee -a $logfile &
