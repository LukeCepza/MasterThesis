import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 

def read_ans(filename):
    # Read the content of the answer text file and store it as a list of lines
    with open(filename, 'r') as file:
        answer_lines = file.readlines()
    print(len(answer_lines))
    a = np.zeros((len(answer_lines),4))
    i = 0
    for line in answer_lines:
        a[i, :] = [int(char) for char in line.strip()]  # Convert each character to an integer
        i += 1  # Increment the row index
    da = np.diff(a)
    da[da > 1] = 1
    da[da < 1] = -1
    return [a,da]

def score_IME(filename,a,da):
    # Read the content of the answer text file and store it as a list of lines
    with open(filename, 'r') as file:
        response = file.readlines()
    r = np.zeros((len(response),4))
    i = 0
    for line in response:
        r[i, :] = [int(char) for char in line.strip()]  # Convert each character to an integer
        i += 1  # Increment the row index
    dr = np.diff(r)
    dr_er = da - dr
    r_er = a - r
    return [dr_er,r_er]

def score(filename,a,da):
    # Read the content of the answer text file and store it as a list of lines
    with open(filename, 'r') as file:
        response = file.readlines()
    r = np.zeros((len(response),4))
    i = 0
    for line in response:
        r[i, :] = [int(char) for char in line.strip()]  # Convert each character to an integer
        i += 1  # Increment the row index
    dr = np.diff(r)
    dr[dr > 1] = 1
    dr[dr < 1] = -1
    r_ex = np.sum(a == r,axis = 1)
    r_chg = np.sum(da == dr,axis = 1)
    return [r_ex,r_chg]

def summary(r_ex,r_chg):
    output = np.zeros((8))
    if len(r_ex) == 60:
        output[0] = np.sum(r_ex[0:19])
        output[1] = np.sum(r_ex[20:39])
        output[2] = np.sum(r_ex[40:59])
        output[4] = np.sum(r_chg[0:19])
        output[5] = np.sum(r_chg[20:39])
        output[6] = np.sum(r_chg[40:59])
    else:
        output[0] = np.sum(r_ex[0:16])
        output[1] = np.sum(r_ex[17:36])
        output[2] = np.sum(r_ex[37:56])
        output[4] = np.sum(r_chg[0:16])
        output[5] = np.sum(r_chg[17:36])
        output[6] = np.sum(r_chg[37:56])
    output[3] = np.sum(r_ex)
    output[7] = np.sum(r_chg)        
    return output

def summary_porcentage(df):
    df['eAir'] = (df['eAir'] / 68) * 100
    df['eVib'] = (df['eVib'] / 80) * 100
    df['eCar'] = (df['eCar'] / 80) * 100
    df['eAll'] = (df['eAll'] / 228) * 100
    df['dAir'] = (df['dAir'] / 51) * 100
    df['dVib'] = (df['dVib'] / 60) * 100
    df['dCar'] = (df['dCar'] / 60) * 100
    df['dAll'] = (df['dAll'] / 171) * 100
    return df

def summary_porcentage_60r(df):
    df['eAir'] = (df['eAir'] / 80)  * 100
    df['eVib'] = (df['eVib'] / 80)  * 100
    df['eCar'] = (df['eCar'] / 80)  * 100
    df['eAll'] = (df['eAll'] / 240) * 100
    df['dAir'] = (df['dAir'] / 60)  * 100
    df['dVib'] = (df['dVib'] / 60)  * 100
    df['dCar'] = (df['dCar'] / 60)  * 100
    df['dAll'] = (df['dAll'] / 180) * 100
    return df
# %%
df = pd.DataFrame(columns=["ID", "eAir", "eVib", "eCar", "eAll", "dAir", "dVib", "dCar", "dAll"])
[answer_razer, answer_diffs_razer] = read_ans("D:\\shared_git\\MaestriaThesis\\data\\ANSWERSHEET_Razer.txt")
for folder_id in range(13,50):
    try:
        if folder_id not in [0,1,2,3,4,5,6,7,8,9,10,11]:
            [r_ex,r_chg] = score(f"D:\\shared_git\\MaestriaThesis\\data\\ID{folder_id:02d}\\R_ID{folder_id:02d}.txt",answer_razer,answer_diffs_razer)
            output = summary(r_ex,r_chg)
            output = np.hstack((folder_id,output.T))
            df.loc[folder_id] = output
    except:
        df.loc[folder_id] = np.zeros(9) 


df = df[(df != 0).all(axis=1)]
df.reset_index(drop=True, inplace=True)
df = summary_porcentage_60r(df)
df.to_csv("Scores_razer.csv")
df.boxplot(column=["eAir", "eVib", "eCar", "eAll", "dAir", "dVib", "dCar", "dAll"])

# Set the title and labels
plt.title("57 Repetitions Razer Headphones")
plt.ylabel("Score (0-100%)")
plt.ylim(0, 100)

# Show the box plot plt.show()
print(df)
output_ex = np.zeros((57,4))
output_chg = np.zeros((57,3))

print(output_ex.shape)
df = pd.DataFrame(columns=["ID", "eAir", "eVib", "eCar", "eAll", "dAir", "dVib", "dCar", "dAll"])
[answer_razer, answer_diffs_razer] = read_ans("D:\\shared_git\\MaestriaThesis\\data\\ANSWERSHEET_Razer.txt")
for folder_id in range(13,50):
    file_path = f"D:\\shared_git\\MaestriaThesis\\data\\ID{folder_id:02d}\\R_ID{folder_id:02d}.txt"
    if os.path.exists(file_path):
        try:
            if folder_id not in [0,1,2,3,4,5,6,7,8,9,10,11]:
                [r_ex,r_chg] = score_IME(file_path,answer_razer,answer_diffs_razer)
                output_ex = np.hstack(( output_ex,r_ex))
                output_chg = np.hstack((output_ex,r_chg))
        except Exception as e:
            print(f'Error while processing: {str(e)}')
    else:
        print(f"The file {file_path} does not exist.")
print(output_ex.shape)