import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow_datasets as tfds
import tool_box
import librosa
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np 
import random

def load_vctk_dataset(batch_size=256, load_all=False):

    '''
        loads train and test datasets that have the following split:

        x => audio vector data
        y => text from transcribed audio

        - optional load_all arg determines how much data should be returned; minimizes loading time
        when set to default False - change to true when ready from complete training session;

        
    '''
    

    ds, info = tfds.load('vctk', with_info=True)
    assert isinstance(ds["train"], tf.data.Dataset), tool_box.color_string('red', f'\n\nINVALID DATASET TYPE RETURNED: received: {type(ds)}\n')

    #vctk dataset only has train split no test or val; create split now
   
    ds = ds["train"].shuffle(buffer_size=1000, seed=42)
    ds = ds.map(lambda d: (d["speech"], d["text"]))
    if load_all == False:
        #set all data to include only 10% of data if load_all set to false
        total_examples = info.splits['train'].num_examples * .10
    else:
        total_examples = info.splits['train'].num_examples
    

    test_split_ratio = 0.2
    test_size = int(test_split_ratio * total_examples)
    train_size = int(info.splits["train"].num_examples - total_examples)
    


    train_data = ds.skip(train_size)
    test_data = ds.take(test_size)

    train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)
    test_data = test_data.prefetch(tf.data.experimental.AUTOTUNE)

    print("Train data:", len(list(train_data)))
    print("Test data:", len(list(test_data)))
    for example in train_data.take(1):
        speech = example[0]
        text = example[1]
        print(f"speech data\n {speech}\ntext data: {text}")
    # return ds



def load_vctk_dataset(batch_size=256, load_all=False):

    '''
        loads train and test datasets that have the following split:

        x => audio vector data
        y => text from transcribed audio

        - optional load_all arg determines how much data should be returned; minimizes loading time
        when set to default False - change to true when ready from complete training session;

        
    '''
@tool_box.timer  
def create_audio_files(sample_size=10000):
    print(tf.config.list_physical_devices('GPU'))

    ds, info = tfds.load('vctk', with_info=True)
    assert isinstance(ds["train"], tf.data.Dataset), tool_box.color_string('red', f'\n\nINVALID DATASET TYPE RETURNED: received: {type(ds)}\n')
  
    train_count = info.splits["train"].num_examples
    count = 0
    ds = ds["train"].take(train_count)
    ds = tfds.as_numpy(ds)
    count = 0
    for example in ds:
        if count == sample_size:
            break
        else:
            id = example["id"].decode("UTF-8")
            audio_data = example["speech"]
            text_data = example["text"].decode("UTF-8")
            data = {"id": id, "audio": audio_data, "text": text_data}
            audio_path = f"{os.getcwd()}/audio_data/{id}.pkl"
            tool_box.Create_Pkl(audio_path, audio_data)
            print(f"{data} {count}")
            count += 1 
    # while count < sample_size:
    #     print(type(ds))
    #     count += 1
        


def create_mel_spectrogram(audio_path, save_path):
    try:

        audio_data = tool_box.Load_Pkl(audio_path).astype("float32")
        sample_rate = 22050

        S = np.abs(librosa.stft(audio_data, n_fft=4096))**2
        chroma = librosa.feature.chroma_stft(S=S, sr=sample_rate)
        S_db = librosa.power_to_db(S=chroma, ref=np.max)
        img = librosa.display.specshow(S_db, sr=sample_rate, fmax=8000)

        plt.savefig(save_path, transparent = True, pad_inches = 0, bbox_inches="tight")
        plt.close()
        print(tool_box.color_string('green',f"\nADDED: {save_path}\n"))
        return True
    
    except Exception as e:
        print(tool_box.color_string('red', f"\nFAILED: {audio_path} error: {e}\n"))
        return False

def load_image(path, resize_dims=(512,512)):

    # load images
    loaded = Image.open(path)
    loaded = loaded.resize(resize_dims)
 
    return loaded

def resize_all_images():

    def check_results(good_images, img_paths):
        user_input_prompt = tool_box.color_string('yellow', f"\n\nALL FILES SUCCESSFULLY RESIZED; DELETE ORIGINAL DATASET AND SET NEW DATASET TO RESIZED?\n ('y' or 'n')\n\n")
        resp = input(user_input_prompt)

        if resp.lower() == 'y':
            #handle rename of temp_images path to image_data and deletion of old
            os.system(f"rm -rf {os.getcwd()}/image_data")
            os.rename(f"{os.getcwd()}/temp_images", f"{os.getcwd()}/image_data")
            print(tool_box.color_string('green', f'\nREMOVED {f"{os.getcwd()}/image_data"} + RENAMED TEMP DIR\n'))
            return True
        
        elif resp.lower() == 'n':
            print(f"\nSKIPPING REMOVAL OF ORIGINAL IMAGES\n")
            return True
        
        else:
            msg = tool_box.color_string('red', f"\nINVALID RESPONSE ENTERED; use 'n' or 'y' only\n")
            print(msg)
            return check_results(good_images, img_paths)
        

    if os.path.exists(f"{os.getcwd()}/temp_images") == True:
        print(tool_box.color_string('red', f"\nTEMP IMAGE FILE EXISTS ALREADY; DELETE OR CHANGE NAME TO CONTINUE\n"))
        return None
    else:
        print(f"\nRESIZING IMAGES...\n")
        os.mkdir(f"{os.getcwd()}/temp_images")
        image_paths = [f"{os.getcwd()}/image_data/{f}" for f in os.listdir("image_data")]
        resized_image_paths = [f"{os.getcwd()}/temp_images/{f}".strip("_ORIGINAL.png") + "_resized.png" for f in os.listdir("image_data")]
        good = []
        bad = []
        count = 0
        for i in range(len(image_paths)):
            try:
                    
                original_path = image_paths[i]
                resize_path = resized_image_paths[i]
                resized_original = load_image(original_path)
                resized_original.save(resize_path)
                width, height = resized_original.size
                msg = tool_box.color_string('green', f'\nADDED: {resize_path} from {original_path}; new dims: ({width}, {height})\t') + tool_box.color_string('yellow', f'({count+1} out of {len(image_paths)})\n')
                print(msg)
             
                good.append(original_path)
                count += 1
            except:
                msg = tool_box.color_string('red', f'\nFAILED: {original_path}\t') + tool_box.color_string('yellow', f'({count+1} out of {len(image_paths)})\n')
                print(msg)
                bad.append(original_path)
                count += 1


        if len(good) == len(image_paths):
            check_results(good, image_paths)

        else:
            tool_box.Create_Pkl("failed_resize_images.pkl", bad)
            failed_msg = tool_box.color_string('yellow', f'\n\nDONE RESIZING FILES; FAILED IMAGES: {len(bad)}\nSEE failed_resize_images.pkl for list of paths\n')

            print(failed_msg)
            return None
        

def create_all_image_data():
    print(f"\nCHECKING PATHS...\n")
    audio_paths = [f"{os.getcwd()}/audio_data/{f}" for f in os.listdir("audio_data") 
                   if f.split("/")[-1].split(".pkl")[0] + "_ORIGINAL" + ".png" not in os.listdir(f"{os.getcwd()}/image_data") and 
                   f.split("/")[-1].split(".pkl")[0] + "_resized" + ".png" not in os.listdir(f"{os.getcwd()}/image_data")]
    good = 0
    bad = 0
    
    print(f"\nADDING {len(audio_paths)} images...\n")
    try:

        for i in range(len(audio_paths)):
            audio_path = audio_paths[i]
            save_path = os.getcwd() + "/image_data/" + audio_path.split("/")[-1].split(".pkl")[0] + "_ORIGINAL" + ".png"
    
            img = create_mel_spectrogram(audio_path, save_path)
            if img == True:
                print(f"GOOD: {good+1}\tBAD: {bad} ({i+1} out of {len(audio_paths)})\n\n")
                good += 1
            else:
                print(f"GOOD: {good}\tBAD: {bad+1} ({i+1} out of {len(audio_paths)})\n")
                bad += 1
    except:
        return create_all_image_data()

    return resize_all_images()


def get_all_image_sizes():
    image_paths = [f"{os.getcwd()}/image_data/{f}" for f in os.listdir("image_data")]
    for i in range(len(image_paths)):
        loaded = load_image(image_paths[i])
        width, height = loaded.size
        print(tool_box.color_string('green',f"\nIMAGE {image_paths[i]} SIZE: ({width}, {height})\n"))






@tool_box.timer
def create_training_dirs(test_split=0.1):
    top_dir = f"{os.getcwd()}/train_session_data"
    #check if train_session exists and delete to create new one for session
    if os.path.exists(top_dir):
        os.system(f"rm -rf {top_dir}")
        
    
    os.mkdir(top_dir)
    dirs = [f"{top_dir}/train",f"{top_dir}/test"]#, f"{top_dir}/val"]
    # for dir in dirs:
    #     os.mkdir(dir)

    #     os.mkdir(f"{dir}/male")
    #     os.mkdir(f"{dir}/female")
    os.mkdir(f"{top_dir}/male")
    os.mkdir(f"{top_dir}/female")

    print(tf.config.list_physical_devices('GPU'))
    current_paths = [f"{os.getcwd()}/image_data/{f}" for f in os.listdir(f"{os.getcwd()}/image_data")]

    random.shuffle(current_paths)

    current_ids = [f.split("_resized")[0] for f in os.listdir(f"{os.getcwd()}/image_data")]
    
    test_paths = current_paths[:int(len(current_paths) * test_split)]
    train_paths = current_paths[int(len(current_paths) * test_split):]
    current_train_ids = [f.split("_resized")[0].split("/")[-1] for f in train_paths]
    current_test_ids = [f.split("_resized")[0].split("/")[-1] for f in test_paths]
    
    ds, info = tfds.load('vctk', with_info=True)
    assert isinstance(ds["train"], tf.data.Dataset), tool_box.color_string('red', f'\n\nINVALID DATASET TYPE RETURNED: received: {type(ds)}\n')
    train_count = info.splits["train"].num_examples
    count = 0
    ds = ds["train"].take(train_count)
    ds = tfds.as_numpy(ds)
    current_ids = {f.split("_resized")[0]: None for f in os.listdir(f"{os.getcwd()}/image_data")}
    target_count = len(current_ids.keys())
    count = 0
    train_count = 0
    test_count = 0
 
    for example in ds:
        if count == target_count:
            break
        else:

            example_id = example["id"].decode("UTF-8")
    
            example_class = example["gender"] #
            if example_class == 0:
                example_class = "male"
            else:
                example_class = "female"
            
            if example_id in current_train_ids:
                index = current_train_ids.index(example_id)
                current_path = train_paths[index]
                #new_path = f"{top_dir}/train/{example_class}/{example_id}.png"
                new_path = f"{top_dir}/{example_class}/{example_id}.png"
               # print(f"id: {example_id}; path: {current_path}, new_path: {new_path}")
                cmd = f"cp {current_path} {new_path}"
                os.system(cmd)
                count += 1
                train_count += 1
                print(tool_box.color_string('green', f'\nADDED {count} out of {target_count}\n'))

            elif example_id in current_test_ids:
                index = current_test_ids.index(example_id)
                current_path = test_paths[index]
                #new_path = f"{top_dir}/test/{example_class}/{example_id}.png"
                new_path = f"{top_dir}/{example_class}/{example_id}.png"
               # print(f"id: {example_id}; path: {current_path}; new_path: {new_path}")
                cmd = f"cp {current_path} {new_path}"
                os.system(cmd)
                count += 1
                test_count += 1
                print(tool_box.color_string('green', f'\nADDED {count} out of {target_count}\n'))

            else:
              #  print(example_id)
                pass
    
    print("test ", test_count)
    print("train ", train_count)
    total_count = train_count + test_count
    print("found count " , total_count )

