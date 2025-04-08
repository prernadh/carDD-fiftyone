import fiftyone as fo
import fiftyone.utils.labels as fol
import fiftyone.brain as fob

from ultralytics import YOLO
from fiftyone.utils.huggingface import push_to_hub, load_from_hub
import fiftyone.utils.image as foui

def load_test_data():
    data_path = "/Users/prerna/datasets/CarDD_release/CarDD_COCO/test2017"
    labels_path = "/Users/prerna/datasets/CarDD_release/CarDD_COCO/annotations/instances_test2017.json"

    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=data_path,
        labels_path=labels_path,
    )
    dataset.name = 'carDD-test'
    dataset.persistent = True

def load_train_data():
    data_path = "/Users/prerna/datasets/CarDD_release/CarDD_COCO/train2017"
    labels_path = "/Users/prerna/datasets/CarDD_release/CarDD_COCO/annotations/instances_train2017.json"

    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=data_path,
        labels_path=labels_path,
    )
    dataset.name = 'carDD-train'
    dataset.persistent = True

def load_val_data():
    data_path = "/Users/prerna/datasets/CarDD_release/CarDD_COCO/val2017"
    labels_path = "/Users/prerna/datasets/CarDD_release/CarDD_COCO/annotations/instances_val2017.json"

    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=data_path,
        labels_path=labels_path,
    )
    dataset.name = 'carDD-val'
    dataset.persistent = True

def export_to_yolo(dataset, split, classes = ["crack", "dent", "scratch", "glass shatter", "tire flat", "lamp broken"]):
    export_path = f"/Users/prerna/datasets/carDD"
    fol.instances_to_polylines(dataset, "segmentations", "polylines")
    label_field = "polylines"
    dataset.export(
        export_dir=export_path,
        dataset_type=fo.types.YOLOv5Dataset,
        split=split,
        label_field=label_field,
        classes=classes,
    )

def compute_thumbnails(dataset, img_size = 512):
    # Generate some thumbnail images
    foui.transform_images(
        dataset,
        size=(-1, img_size),
        output_field="thumbnail_path",
        output_dir="/tmp/thumbnails",
    )

    dataset.app_config.media_fields = ["filepath", "thumbnail_path"]
    dataset.app_config.grid_media_field = "thumbnail_path"
    dataset.save()  # must save after edits

def compute_emb_visualization(dataset):
    # You can select any 
    fob.compute_visualization(
        dataset,
        model="clip-vit-base32-torch",
        brain_key="clip_viz",
        create_index=True,
        points_field="clip_viz_embeddings"
    )

    fob.compute_visualization(
        dataset,
        model="clip-vit-base32-torch",
        patches_field="ground_truth",
        brain_key="clip_viz_patches",
    )

def compute_dataset_similarity(dataset):
    results = fob.compute_similarity(
        dataset,
        model="clip-vit-base32-torch",
        backend="sklearn",  # "sklearn", "qdrant", "redis", etc
        brain_key="img_sim",
    )
    # Make the brain index also sort by text similarity here

def uniqueness_representativeness(dataset):
    fob.compute_uniqueness(dataset)
    fob.compute_representativeness(dataset)

def compute_near_duplicates(dataset):
    index = fob.compute_near_duplicates(dataset)
    dups_view = index.duplicates_view()
    dataset.save_view("near-duplicates", dups_view)

def train():
    model = YOLO('yolo11s-seg.pt')  # Load in the pre-trained YOLOv11 segmentation model

    # Set up training configuration
    model.train(
        data='/Users/prerna/datasets/carDD/dataset.yaml',  
        epochs=100,                                        
        batch=8,                                          
        imgsz=640,                                         
        val=True,                                          
        save=True,                                         
        save_period=5,                                     
        project='car_damage_segmentation',                 
        name='yolov11_seg_run',                            
        workers=4,                                         
        device='mps',                                      
        patience=10, #Number of epochs with no improvement after which training will be stopped                                      
        verbose=True                                       
    )

def apply_model(dataset):
    model_path = "/Users/prerna/models/carDD/best.pt"
    model = YOLO(model_path)

    dataset = fo.load_dataset("carDD-test")
    dataset.apply_model(model, label_field="yolo11_large")

def evaluate_model(dataset):
    fo.evaluate_detections(dataset, pred_field="yolo11_small", gt_field = "ground_truth", eval_key = "yolo11_small_eval", use_masks = True, compute_mAP=True)
    fo.evaluate_detections(dataset, pred_field="yolo11_large", gt_field = "ground_truth", eval_key = "yolo11_large_eval", use_masks = True, compute_mAP=True)

def push_to_hub(dataset, name):
    push_to_hub(dataset, name, token = "###")

if __name__ == "__main__":
    train()