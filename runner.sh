CUDA_VISIBLE_DEVICES="5" python3 train_continual.py --labels_to_learn 3 --ckpt_folder /home/stud-1/aditya/vae/results/mnist/2024_03_26_024855

CUDA_VISIBLE_DEVICES="5" python calculate_fim.py --ckpt_folder /home/stud-1/aditya/vae/results/mnist/2024_03_26_024855

CUDA_VISIBLE_DEVICES="5" python train_cvae_partial.py --config mnist.yaml --data_path ./dataset --labels_to_learn 1 2 

python train_classifier.py --data_path ./dataset

python final.py --config mnist.yaml --data_path ./dataset --input_file input_file2.txt

python final.py --config mnist.yaml --data_path ./dataset --input_file input_file2.txt --n_passes 20
# continual learning is without EWC
# forgetting is with EWC