import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from cad_utils import render_cad_views


def process_cad_folder(cad_folder_path):
    cad_folder_path = cad_folder_path.strip()
    output_dir = f'{cad_folder_path}/mesh_views'
    os.makedirs(output_dir, exist_ok=True)

    # check if some files already exist in output_dir, if so, skip rendering
    if len(os.listdir(output_dir)) > 0 and 0:
        return f"Already rendered: {cad_folder_path}"
    else:
        print(f"Rendering: {cad_folder_path}")
    
        cad_file_path = None
        for file in os.listdir(cad_folder_path):
            if file.endswith('.obj'):
                mesh_file_path = os.path.join(cad_folder_path, file)
                break

        if mesh_file_path is None:
            return f"No OBJ file in {cad_folder_path}"

        # render_cad_views(
        #     cad_file_path,
        #     output_dir=output_dir,
        #     n_azimuth=12,
        #     n_elevation=5,
        #     verbose=False,
        # )

        render_cad_views(
            mesh_file_path,
            output_dir=output_dir,
            n_azimuth=12,
            n_elevation=5,
            verbose=False,
            use_cad=False
        )
        

        return f"Done: {cad_folder_path}"


if __name__ == '__main__':

    for split in ['val_new']:
        dataset_path = f'/data/1bali/Other_LLM_projects/ECCV_2026/LISA/{split}_dataset.log'

        with open(dataset_path) as fread:
            lines = fread.readlines()

        # 🔥 Adjust this depending on your CPU cores
        num_workers = 150

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_cad_folder, line) for line in lines]

            for _ in tqdm(as_completed(futures), total=len(futures)):
                pass