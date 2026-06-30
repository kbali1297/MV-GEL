import os
# Set your OpenRouter key in the environment: export OPENROUTER_API_KEY=...
api_key = os.environ.get('OPENROUTER_API_KEY', '')
import argparse
import base64
import requests
from tqdm import tqdm
import multiprocessing as mp
import builtins
import sys
import random
from cad_utils import compute_optimal_views, render_cad_views
import time

# ----------------------------
# Queue print helper
# ----------------------------
def qprint(queue, *args):
    queue.put(" ".join(str(a) for a in args))


# ----------------------------
# Log listener (MAIN PROCESS)
# ----------------------------
def log_listener(queue):
    while True:
        msg = queue.get()
        if msg == "__STOP__":
            break
        print(msg, flush=True)



def safe_openrouter_call(payload, headers, max_retries=5):
    for attempt in range(max_retries):
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
            r.raise_for_status()
            return r.json()

        except requests.exceptions.HTTPError as e:
            if r.status_code in {429, 500, 502, 503, 504}:
                time.sleep(2 ** attempt + random.random())
            else:
                raise

        except requests.exceptions.RequestException:
            time.sleep(2 ** attempt + random.random())

    raise RuntimeError("OpenRouter failed after retries")

# ----------------------------
# Worker function (TOP-LEVEL!)
# ----------------------------
def process_one_cad(job):
    """
    job = (cad_folderpath, dataset_path, llm_name, api_key, log_queue)
    """

    cad_folderpath, dataset_path, llm_name, api_key, log_queue = job
    cad_dir = os.path.join(dataset_path, cad_folderpath)

    # ------------------------------------------------
    # Redirect *all* prints in this process to queue
    # ------------------------------------------------
    builtins.print = lambda *a, **k: qprint(log_queue, *a)

    try:
        # ----------------------------
        # Find STEP file
        # ----------------------------
        cad_filepath = None
        for file in os.listdir(cad_dir):
            if file.endswith(".step"):
                cad_filepath = os.path.join(cad_dir, file)
                break

        if cad_filepath is None:
            print(f"[SKIP] {cad_folderpath}: no STEP file")
            return

        print(f"[{cad_folderpath}] Using CAD:", cad_filepath)

        # ----------------------------
        # Compute optimal views for all top edges and surfaces
        # ----------------------------
        edge_viewpaths_list, face_viewpaths_list, edge_idxs, face_idxs = compute_optimal_views(
            cad_filepath,
            n_azimuth=12,
            n_elevation=5,
            num_top_edges=5,
            num_top_faces=5,
            orthographic=False,
        )
        # except Exception as e:
        #     print(f"[FAIL] {cad_folderpath}:", e)
        #     return

        for i, (edge_viewpaths, face_viewpaths) in enumerate(zip(edge_viewpaths_list, face_viewpaths_list)): 
            
            #top views of the respective edge and face, later iterations would include other views also
            edge_viewpath, face_viewpath = random.choice(edge_viewpaths[:5]), random.choice(face_viewpaths[:5]) 

            print(f"[{cad_folderpath}] {i}th Edge view:", edge_viewpath)
            print(f"[{cad_folderpath}] {i}th Face view:", face_viewpath)

            # Parse angles
            el_edge = int(os.path.basename(edge_viewpath).split("_e")[1].split("_")[0])
            az_edge = int(os.path.basename(edge_viewpath).split("_a")[1].split("_")[0])

            el_face = int(os.path.basename(face_viewpath).split("_e")[1].split("_")[0])
            az_face = int(os.path.basename(face_viewpath).split("_a")[1].split("_")[0])

            # ----------------------------
            # Render views (VERBOSE WORKS)
            # ----------------------------
            render_cad_views(
                cad_filepath,
                n_azimuth=[az_edge, az_face],
                n_elevation=[el_edge, el_face],
                orthographic=False,
                verbose=True,
            )

            render_cad_views(
                cad_filepath,
                n_azimuth=[az_edge],
                n_elevation=[el_edge],
                orthographic=False,
                highlight_edge_idxs=edge_idxs[i:i+1],
                verbose=True,
            )

            render_cad_views(
                cad_filepath,
                n_azimuth=[az_face],
                n_elevation=[el_face],
                orthographic=False,
                highlight_face_idxs=face_idxs[i:i+1],
                verbose=True,
            )

            # ----------------------------
            # Prompts
            # ----------------------------

            generate_question_prompt = (
            "You are generating a spatial grounding question for a 3D CAD understanding task.\n\n"
            "You are given exactly TWO images of the SAME CAD part, provided in this order:\n"
            "1) An unannotated 3D rendering of the part.\n"
            "2) A second rendering where EITHER:\n"
            "   - a single surface is marked, OR\n"
            "   - a single edge is marked.\n\n"
            "The model that will later answer this question will see ONLY the unannotated image.\n\n"
            "Your task:\n"
            "Generate ONE concise question that allows the answering model to uniquely identify "
            "the marked entity (surface or edge) using ONLY geometric and topological cues visible "
            "in the unannotated image.\n\n"
            "Entity rules:\n"
            "- If the marked entity is a surface, refer ONLY to surfaces.\n"
            "- If the marked entity is an edge, refer ONLY to edges.\n"
            "- Do NOT mix surfaces and edges in the description.\n\n"
            "Description constraints:\n"
            "- Do NOT mention colors, markings, highlights, annotations, or how the entity was indicated.\n"
            "- Do NOT refer to multiple images or comparisons between images.\n"
            "- Do NOT use perspective-specific language (e.g., camera view, zoom, rendering style).\n"
            "Preferred descriptors:\n"
            "- Position relative to the part (axial, radial, proximal/distal).\n"
            "- Orientation inferred from the visible coordinate axes.\n"
            "- Adjacency to other surfaces or edges.\n"
            "- Curvature, convexity/concavity, continuity.\n"
            "- Symmetry, repetition, or termination at joints or features.\n\n"
            "Output rules:\n"
            "- Output ONLY the question.\n"
            "- The question must be sufficient to uniquely identify the entity.\n"
            "- Do NOT add explanations, labels, or extra text.\n\n"
            "Generate the question now."
            )

            # ----------------------------
            # Edge + Face inference
            # ----------------------------
            for feature, viewpath, feature_idx in zip(
                ["edge", "face"],
                [edge_viewpath, face_viewpath],
                [edge_idxs, face_idxs]
            ):
                img_paths = [
                    viewpath.replace(f"_marked_{feature}.png", ".png"),
                    viewpath.replace(f"_marked_{feature}.png", f"_marked_{feature}{feature_idx[i:i+1]}.png"),
                ]

                print(f"[{cad_folderpath}] Inference images:", img_paths)

                #-----------------------------Generate Question ---------------------------------
                inference_imgs_1 = []
                for img_path in img_paths:
                    with open(img_path, "rb") as f:
                        inference_imgs_1.append(
                            base64.b64encode(f.read()).decode("utf-8")
                        )

                messages_1 = [{
                    "role": "user",
                    "content": [{"type": "text", "text":generate_question_prompt}] + [
                        {
                            "type": "image_url",
                            "image_url": f"data:image/png;base64,{img}",
                        }
                        for img in inference_imgs_1
                    ],
                }]

                response_json_1 = safe_openrouter_call(
                    payload={
                        "model": llm_name,
                        "temperature": 0.0,
                        "messages": messages_1,  # <-- YOUR PROMPT UNCHANGED
                    },
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                )

                question = response_json_1["choices"][0]["message"]["content"]
                # ---------------------- Generate answer to question ---------------------------
                answer_prompt = f"{question} Please make the answer brief"
                inference_imgs_2 = []
                for img_path in img_paths[:1]: # For inference the answering model sees only the first unannotated image
                    with open(img_path, "rb") as f:
                        inference_imgs_2.append(
                            base64.b64encode(f.read()).decode("utf-8")
                        )

                messages_2 = [{
                    "role": "user",
                    "content": [{"type": "text", "text":answer_prompt}] + [
                        {
                            "type": "image_url",
                            "image_url": f"data:image/png;base64,{img}",
                        }
                        for img in inference_imgs_2
                    ],
                }]
                
                response_json_2 = safe_openrouter_call(
                    payload={
                        "model": llm_name,
                        "temperature": 0.0,
                        "messages": messages_2,  # <-- YOUR PROMPT UNCHANGED
                    },
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                )

                
                answer = response_json_2["choices"][0]["message"]["content"]
                
                out_path = os.path.join(
                    cad_dir,
                    f"views_and_ques_{feature}.log",
                )

                with open(out_path, "a") as f:
                    f.write(
                        str({f"{feature}_idx": i,
                            "unmarked_image": img_paths[0],
                            "marked_image": img_paths[1],
                            "question": question,
                            "answer": answer,
                            "top_views_desc": edge_viewpaths if feature=='edge' else face_viewpaths
                        })
                    )
                    f.write("\n")

                print(f"[{cad_folderpath}] Saved:", out_path)

            print(f"[OK] {cad_folderpath}")

    except Exception as e:
        print(f"[FAIL] {cad_folderpath}:", repr(e))
    return      

        


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":

    random.seed(42)
    # ---- SAFETY FOR FREECAD ----
    mp.set_start_method("forkserver", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--LLM_name", type=str, default="openai/gpt-5.1")
    parser.add_argument("--num_workers", type=int, default=24)
    args = parser.parse_args()

    #api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key is None:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    dataset_path = os.environ.get(
        "MVGEL_CAD_DATASET",
        os.path.join(
            os.path.dirname(os.environ.get(
                "MVGEL_ROOT", os.path.dirname(os.path.abspath(__file__)))),
            "ABC_CAD_Dataset_small2"))
    cad_folders = sorted(os.listdir(dataset_path))

    manager = mp.Manager()
    log_queue = manager.Queue()

    # ---- Start log listener ----
    listener = mp.Process(target=log_listener, args=(log_queue,))
    listener.start()

    jobs = [
        (cad, dataset_path, args.LLM_name, api_key, log_queue)
        for cad in cad_folders
    ]

    num_workers = min(args.num_workers, mp.cpu_count())

    with mp.Pool(processes=num_workers) as pool:
        list(
            tqdm(
                pool.imap_unordered(process_one_cad, jobs),
                total=len(jobs),
            )
        )

    #process_one_cad(jobs[1])
    # ---- Stop logger ----
    log_queue.put("__STOP__")
    listener.join()
