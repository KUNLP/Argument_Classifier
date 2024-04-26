
학습 실행 방법

    python run.py --train_file TRAIN_FILE_PATH --save_dir SAVE_DIRECTORY_NAME --do_train True --init_weight True

테스트 실행 방법

    python run.py --predict_file PREDICT_FILE_PATH --output_dir MODEL_DIRECTORY_NAME --checkpoint MODEL_CHECKPOINT --do_eval True

실제 예시

    python run.py --predict_file extractive_summary_mrc_test_4.0.json --output_dir ./ --checkpoint 16000 --do_eval True
    
--output_dir : 저장된 모델을 불러오는 디렉토리. --checkpoint와 같이 엮임.

ex)
--output_dir : ./ 
--checkpoint : 16000
./checkpoint-16000 안에 들어있는 모델 불러옴
