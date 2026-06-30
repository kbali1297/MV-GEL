import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from cad_utils import *
import ast
import shutil

# Repo-relative data root. Dataset logs ({split}_dataset.log) are git-ignored;
# override the location with the MVGEL_ROOT environment variable.
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("MVGEL_ROOT", ROOT)

def process_cad_folder(cad_folder_path):
    cad_folder_path = cad_folder_path.strip()
    output_dir = f'{cad_folder_path}/target_CDviews'#f'{cad_folder_path}/deep_views'
    os.makedirs(output_dir, exist_ok=True)

    feature_lines = []
    with open(f'{cad_folder_path}/views_and_ques_edge_augmented_.log', 'r') as fedge:
        feature_lines.extend(fedge.readlines())
    
    with open(f'{cad_folder_path}/views_and_ques_face_augmented_.log', 'r') as fface:
        feature_lines.extend(fface.readlines())

    #check if some files already exist in output_dir, if so, skip rendering
    # if len(os.listdir(output_dir)) == 60 and 0:
    #     return f"Already rendered: {cad_folder_path}"
    # else:
    print(f"Rendering: {cad_folder_path}")

    
    for line in feature_lines:
        dict_ = ast.literal_eval(line.strip())
        feature, feature_idx = dict_["marked_image"].split('marked_')[1].split('[')[0], int(dict_["marked_image"].split('[')[1].split(']')[0])
        
        cad_file_path = None
        for file in os.listdir(cad_folder_path):
            if file.endswith('.obj'):
                mesh_file_path = os.path.join(cad_folder_path, file)
                break

        if mesh_file_path is None:
            return f"No OBJ file in {cad_folder_path}"

        # render_depth_maps(
        #     mesh_file_path,
        #     output_dir=output_dir,
        #     n_azimuth=12,
        #     n_elevation=5,
        # )

        # render_depth_normal_maps(
        #     mesh_file_path,
        #     output_dir=output_dir,
        #     n_azimuth=12,
        #     n_elevation=5,
        # )

        # render_sobel_edge_maps(
        #     mesh_file_path,
        #     output_dir,
        #     n_azimuth=12,
        #     n_elevation=5,
        # )

        render_cad_views(
            cad_file=mesh_file_path.replace(".obj", ".step"),
            output_dir=f"{cad_folder_path}/target_CDviews",
            n_azimuth=12,#[az],
            n_elevation=5,#[el],
            verbose=False,
            use_cad=True,
            highlight_edge_idxs=[feature_idx] if feature=='edge' else None,
            highlight_face_idxs=[feature_idx] if feature=='face' else None,
            save_img=True)

    return f"Done: {cad_folder_path}"


if __name__ == '__main__':

    for split in ['val']:
        dataset_path = os.path.join(DATA_ROOT, f'{split}_dataset.log')

        with open(dataset_path) as fread:
            lines = fread.readlines()

        # for line in lines:
        #     line = '/data/1bali/Other_LLM_projects/ECCV_2026/ABC_CAD_Dataset_small2/00290064'
        #     process_cad_folder(line.strip())
        #🔥 Adjust this depending on your CPU cores
        num_workers = 150

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_cad_folder, line) for line in lines]

            for _ in tqdm(as_completed(futures), total=len(futures)):
                pass