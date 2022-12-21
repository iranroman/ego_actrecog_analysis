from datasets.epickitchens import Epickitchens
import yaml
import torch
import csv
from dotmap import DotMap

def main(cfg, model):

    # Device on which to run the model
    # Set to cuda to load on GPU
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    # load model

    # trying out the data loader
    with open(cfg,'r') as f:
        temp = yaml.safe_load(f)
    d = DotMap(temp)
    d = Epickitchens(d,'test',model)

    if model == 'omnivore':
        # Pick a pretrained model 
        model_name = "omnivore_swinB_epic"
        model = torch.hub.load("facebookresearch/omnivore:main", model=model_name, force_reload=True)

    # Set to eval mode and move to desired device
    model = model.to(device)
    model = model.eval()

    # Pass the input clip through the model 
    with torch.no_grad():
        for i,(frames,action_index,verb_index,noun_index,metadata) in enumerate(d):
            video_input = frames[None,...]
            prediction = model(video_input.to(device), input_type="video")
            
            # Get the predicted classes 
            pred_classes = prediction.topk(k=5).indices

            print(action_index,pred_classes[0])
            print(pred_classes[0][0] in verb_index)
            print(pred_classes[0][0] in noun_index)
            print()

            input()


if __name__ == "__main__":
    import fire
    fire.Fire(main)
