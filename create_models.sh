if [ ! -d "models" ]; then
    mkdir "models"
fi

python3 timm_train_vit/create_model.py models/vit_1.cvit models/vit_2.cvit models/vit_1.pt models/vit_2.pt && echo models created

if [ ! -d "logs" ]; then
    mkdir "logs"
fi
if [ ! -f "logs/model_info.txt" ]; then
    touch logs/model_info.txt
fi
echo "" >logs/model_info.txt
python3 timm_train_vit/print_model_info.py models/vit_1.cvit logs/model_info.txt
python3 timm_train_vit/print_model_info.py models/vit_2.cvit logs/model_info.txt
echo model info printed on file logs/model_info.txt
