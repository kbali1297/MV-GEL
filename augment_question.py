import os
import ast
import random
from utils.dataset_ import extract_el_az_from_view_desc
from tqdm import tqdm

# def inject_phrase(question, feature, phrase, append_str=""):
#     return question.replace(
#         feature,
#         f"{feature}, {phrase}",
#         1
#     ) + " " + append_str

def fuse_view_phrases(az_phrase, el_phrase):
    az_core = az_phrase.rstrip(",")
    el_core = el_phrase.rstrip(",")

    # Lowercase elevation for smooth fusion
    el_core = el_core[0].lower() + el_core[1:]

    return f"{az_core} and {el_core},"

def inject_view_phrase(question, feature, fused_phrase):
    r = random.random()

    # Leading
    if r < 0.50:
        return f"{fused_phrase} {question}"

    # Mid-clause
    elif r < 0.55:
        return question.replace(
            feature,
            f"{feature}, {fused_phrase.lower()}",
            1
        )

    # # Trailing
    # elif r < 0.80:
    #     return f"{question} {fused_phrase}"

    # Contextual
    else:
        return f"{question} This is most clearly seen {fused_phrase.lower()[:-1]}."

ABOVE_PHRASES = [
    "viewed from above,",
    "seen from above,",
    "as seen from above,",
    "from an elevated viewpoint,",
    "from a top-down perspective,",
    "from a higher vantage point,",
    "observed from above,",
    "looking down at it,",
    "as viewed from an upper angle,",
    "from an overhead angle,",
    "from a superior vertical position,",
    "from a positive elevation angle,",
]

BELOW_PHRASES = [
    "viewed from below,",
    "seen from below,",
    "as seen from below,",
    "from a lower vantage point,",
    "from beneath,",
    "from an underside perspective,",
    "looking up at it,",
    "as viewed from a negative elevation angle,",
    "from an upward-facing perspective,",
    "from a low elevation viewpoint,",
    "from underneath,",
    "with the camera positioned below,",
]

SIDE_PHRASES = [
    "from a lateral viewpoint,",
    "from a horizontal viewing angle,",
    "from a near-zero elevation angle,",
    "with the camera at mid-height,",
    "from a lateral angle,",
    "from an approximately horizontal viewpoint,",
]

FRONT_PHRASES = [
    "Viewed from the front,",
    "Seen from the front,",
    "From a frontal viewpoint,",
    "From the forward-facing side,",
    "From a head-on angle,",
    "As seen from directly in front,",
    "Observed from the front side,",
    "From a straight-on perspective,",
    "From the object's anterior side,",
    "With a direct frontal view,",
    "From the primary front face,",
    "Looking straight at the front,",
]

FRONT_RIGHT_PHRASES = [
    "Viewed from the front-right,",
    "Seen from the front-right side,",
    "From a diagonal front-right angle,",
    "From a front-right perspective,",
    "As seen from the forward-right direction,",
    "Observed from a right-offset frontal position,",
    "From a slightly rotated front-right view,",
    "From an oblique angle toward the front-right,",
    "With the viewpoint shifted toward the front-right,",
    "From the right side of the front face,",
    "From a forward-right orientation,",
    "Looking at it from the front-right corner,",
]

RIGHT_PHRASES = [
    "Viewed from the right side,",
    "Seen from the right,",
    "From the right profile,",
    "From a lateral right angle,",
    "As observed from the right-hand side,",
    "From a direct right-facing perspective,",
    "From the object's right flank,",
    "With a side view from the right,",
    "From a pure rightward orientation,",
    "Looking at it from the right side,",
    "From a strict right profile view,",
    "Observed from directly to the right,",
]

BACK_RIGHT_PHRASES = [
    "Viewed from the back-right,",
    "Seen from the rear-right side,",
    "From a diagonal rear-right angle,",
    "From behind and to the right,",
    "As seen from the rear-right direction,",
    "Observed from a right-offset rear position,",
    "From a rotated back-right viewpoint,",
    "From an oblique angle toward the back-right,",
    "With the viewpoint shifted toward the rear-right,",
    "From the right side of the rear face,",
    "From a backward-right orientation,",
    "Looking at it from the back-right corner,",
]

BACK_PHRASES = [
    "Viewed from the back,",
    "Seen from the rear,",
    "From a rear-facing perspective,",
    "From directly behind,",
    "As observed from the backside,",
    "From the object's posterior side,",
    "With a straight rear view,",
    "From a head-on rear angle,",
    "From the primary back face,",
    "Looking straight at the rear,",
    "Observed from a direct backward orientation,",
    "From the full rear profile,",
]

BACK_LEFT_PHRASES = [
    "Viewed from the back-left,",
    "Seen from the rear-left side,",
    "From a diagonal rear-left angle,",
    "From behind and to the left,",
    "As seen from the rear-left direction,",
    "Observed from a left-offset rear position,",
    "From a rotated back-left viewpoint,",
    "From an oblique angle toward the back-left,",
    "With the viewpoint shifted toward the rear-left,",
    "From the left side of the rear face,",
    "From a backward-left orientation,",
    "Looking at it from the back-left corner,",
]

LEFT_PHRASES = [
    "Viewed from the left side,",
    "Seen from the left,",
    "From the left profile,",
    "From a lateral left angle,",
    "As observed from the left-hand side,",
    "From a direct left-facing perspective,",
    "From the object's left flank,",
    "With a side view from the left,",
    "From a pure leftward orientation,",
    "Looking at it from the left side,",
    "From a strict left profile view,",
    "Observed from directly to the left,",
]

FRONT_LEFT_PHRASES = [
    "Viewed from the front-left,",
    "Seen from the front-left side,",
    "From a diagonal front-left angle,",
    "From a front-left perspective,",
    "As seen from the forward-left direction,",
    "Observed from a left-offset frontal position,",
    "From a slightly rotated front-left view,",
    "From an oblique angle toward the front-left,",
    "With the viewpoint shifted toward the front-left,",
    "From the left side of the front face,",
    "From a forward-left orientation,",
    "Looking at it from the front-left corner,",
]

if __name__ == "__main__":
    
    ## Augment question to include view orientation info 
    ## to make the job of view selector easier.
    ## This is a bit hacky but it works for now.
    for split in ["train", "val"]:

        with open(f'/data/1bali/Other_LLM_projects/ECCV_2026/LISA/{split}_dataset.log', 'r') as f:
            lines = f.readlines()
            cad_folder_paths = [line.strip() for line in lines]
    
        for cad_folder_path in tqdm(cad_folder_paths, total=len(cad_folder_paths)):
            for feature in ['edge', 'face']:
                dict_file_path = f'/data/1bali/Other_LLM_projects/ECCV_2026/ABC_CAD_Dataset_small2/{cad_folder_path.split("/")[-1]}/views_and_ques_{feature}.log'
                dict_file_path_new = f'/data/1bali/Other_LLM_projects/ECCV_2026/ABC_CAD_Dataset_small2/{cad_folder_path.split("/")[-1]}/views_and_ques_{feature}_augmented_.log'
                with open(dict_file_path_new, 'w') as fnew:
                    fnew.write('') # clear the file if it already exists
                for line in open(dict_file_path, 'r'):
                    if line.startswith('{'):
                        feature_dict = ast.literal_eval(line.strip())
                        ## Take top 5 views and see which vertical view is more present
                        ## append that to the question to make it easier for the view selector to learn the mapping between question and view orientation.
                        # above_views = 0
                        # below_views = 0
                        # side_views = 0
                        
                        view_path = feature_dict["marked_image"]
                        el, az = extract_el_az_from_view_desc(view_path)
                        if el > 0:
                            elevation_phrase = random.choice(ABOVE_PHRASES)
                        elif el < 0:
                            elevation_phrase = random.choice(BELOW_PHRASES)
                        else:
                            elevation_phrase = random.choice(SIDE_PHRASES)

                        if az==0:
                            azimuthal_phrase = random.choice(FRONT_PHRASES)
                        elif az<90:
                            azimuthal_phrase = random.choice(FRONT_RIGHT_PHRASES)
                        elif az==90:
                            azimuthal_phrase = random.choice(RIGHT_PHRASES)
                        elif az<180:
                            azimuthal_phrase = random.choice(BACK_RIGHT_PHRASES)
                        elif az==180:
                            azimuthal_phrase = random.choice(BACK_PHRASES)
                        elif az<270:
                            azimuthal_phrase = random.choice(BACK_LEFT_PHRASES)
                        elif az==270:
                            azimuthal_phrase = random.choice(LEFT_PHRASES)
                        else:
                            azimuthal_phrase = random.choice(FRONT_LEFT_PHRASES)

                        # if above_views > below_views:
                        #     if above_views > side_views:
                        #         elevation_phrase = random.choice(ABOVE_PHRASES)
                        #     else:
                        #         elevation_phrase = random.choice(SIDE_PHRASES)
                        # elif below_views > above_views:
                        #     if below_views > side_views:
                        #         elevation_phrase = random.choice(BELOW_PHRASES)
                        #     else:
                        #         elevation_phrase = random.choice(SIDE_PHRASES)
                        # else:
                        #     elevation_phrase = random.choice(SIDE_PHRASES)


                        # feature_dict["question"] = inject_phrase(
                        #     feature_dict["question"],
                        #     feature,
                        #     elevation_phrase, feature_dict["answer"]
                        # )
                        fused_phrase = fuse_view_phrases(azimuthal_phrase, elevation_phrase)


                        feature_dict["question"] = inject_view_phrase(
                            feature_dict["answer"], ## overwrite the question as answer is more concise describing the feature, and we want to inject the view phrase right next to the feature in the question for better learning signal, rather than at the beginning or end of the question.
                            feature,
                            fused_phrase
                        )

                        feature_dict["answer"] = ""
                        el, az = extract_el_az_from_view_desc(feature_dict["unmarked_image"])
                        view_path = feature_dict["unmarked_image"].replace('.png', f'_marked_{feature}.png')
                        ## Add this to the top of the dict since the question and answer are based on this view, and we want the model to pay more attention to it.
                        feature_dict["top_views_desc"] = [view_path] + [v for v in feature_dict["top_views_desc"] if v.split("_marked_")[0] != feature_dict["unmarked_image"].split(".png")[0]]
                        with open(dict_file_path_new, 'a') as fnew:
                            fnew.write(str(feature_dict) + '\n')

