from datasets.epickitchens import Epickitchens
import yaml
import torch
import csv
from dotmap import DotMap
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

def main(cfg, model):

    # Device on which to run the model
    # Set to cuda to load on GPU
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    # load model

    # trying out the data loader
    with open(cfg,'r') as f:
        temp = yaml.safe_load(f)
    cfg = DotMap(temp)
    eval_data = Epickitchens(cfg,'test',model)
    eval_loader = DataLoader(eval_data, batch_size = cfg.DATA.BATCH_SIZE, num_workers=cfg.DATA.NUM_WORKERS, shuffle=False)

    if model == 'omnivore':
        # Pick a pretrained model 
        model_name = "omnivore_swinB_epic"
        model = torch.hub.load("facebookresearch/omnivore:main", model=model_name)#, force_reload=True)

    # Set to eval mode and move to desired device
    model = model.to(device)
    model = model.eval()



    # Pass the input clip through the model 
    with torch.no_grad():
        #for kkk,(video_input,action_index,verb_index,noun_index,metadata) in enumerate(tqdm(eval_loader)):
        eval_top1_acc = 0
        for kkk,(video_input,action_index) in enumerate(tqdm(eval_loader)):

            video_input = video_input.to(device)

            prediction = model(video_input.to(device), input_type="video")
            
            # Get the predicted classes 
            pred_classes = prediction.topk(k=5, dim=-1).indices.cpu()

            batch_top1_acc = torch.sum(pred_classes[:,0]==action_index)/len(video_input)
            eval_top1_acc += batch_top1_acc
            if (kkk % 100) == 0:
                print(kkk)
                print('eval accuracy so far:', eval_top1_acc/(kkk+1))

    print('final eval accuracy (top1)', eval_top1_acc/len(eval_loader))


if __name__ == "__main__":
    import fire
    fire.Fire(main)
