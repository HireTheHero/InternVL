DOWNLOAD_DIR=$1
echo $DOWNLOAD_DIR
git clone git@github.com:ronghanghu/gqa_eval_script.git
cp gqa_eval_script/eval.py $DOWNLOAD_DIR
rm -rf gqa_eval_script