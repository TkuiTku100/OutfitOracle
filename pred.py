import csv
import math
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.neighbors import NearestNeighbors
from flask import Flask, jsonify, request, send_from_directory
import pandas as pd
import random
from flask_cors import CORS  # If you're dealing with CORS issues
import os
import matplotlib.colors as mcolors

app = Flask(__name__)
CORS(app)  # Enable CORS if your client is on a different origin

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

@app.route('/wardrobe-select')
def wardrobe_select():
    user = request.args.get('user')
    df = pd.read_csv(f'wardrobe_{user}.csv', header=0)
    category_type = request.args.get('category', 'all')  # Default to 'top' if type not specified
    color_type = request.args.get('color', 'all')
    if category_type!='all':
        df=df[df['Category']==category_type]
    if color_type!='all':
        df=df[df['Color']==color_type]
    return jsonify(df['ImagePath'].tolist())


flag_col = 19
start_of_top = 0
end_of_top = 17
start_of_overtop = 17
end_of_overtop = 34
start_of_bottom = 34
end_of_bottom = 51
no_overtop = "1711917857463.png"


@app.route('/predict-image')
def predict_image():
    type = request.args.get('type', 'generate')
    print(type)
    condition = request.args.get('condition', 'sunny')
    user = request.args.get('user')
    t1=request.args.get('temperature', '20')
    print(t1)
    try:
        temperature=int(t1)
    except:
        temperature=20
    h1 = request.args.get('hour', '20')
    try:
        hour = int(h1)
    except:
        hour = 12
    if type == 'generate':
        return jsonify({"predictedImagePath":predict_outfit(condition, temperature, hour, user)})
    if type == 'regenerate':
        return jsonify({"predictedImagePath":regenerate_outfit(user)})
    if type == 'top':
        return jsonify({"predictedImagePath":change_top(user)})
    if type == 'bottom':
        return jsonify({"predictedImagePath":change_bottom(user)})
    if type == 'overtop':
        return jsonify({"predictedImagePath":change_overtop(user)})

@app.route('/new-user')
def is_new_user():
    # Define file paths
    user = request.args.get('user')
    train_file = f"train_{user}.csv"
    wardrobe_file = f"wardrobe_{user}.csv"

    if not os.path.exists(train_file):
        header = ['sunny', 'cloudy', 'rainy', 'temp', 'time', 'top_Length', 'top_Fit',
                  'top_red', 'top_orange', 'top_yellow', 'top_green', 'top_blue', 'top_purple',
                  'top_black', 'top_white', 'top_gray', 'top_brown', 'top_drawing', 'top_plain',
                  'top_plaid', 'top_horizontal-lines', 'top_vertical-lines', 'overtop_Length',
                  'overtop_Fit', 'overtop_red', 'overtop_orange', 'overtop_yellow', 'overtop_green',
                  'overtop_blue', 'overtop_purple', 'overtop_black', 'overtop_white', 'overtop_gray',
                  'overtop_brown', 'overtop_drawing', 'overtop_plain', 'overtop_plaid',
                  'overtop_horizontal-lines', 'overtop_vertical-lines', 'bottom_Length', 'bottom_Fit',
                  'bottom_red', 'bottom_orange', 'bottom_yellow', 'bottom_green', 'bottom_blue',
                  'bottom_purple', 'bottom_black', 'bottom_white', 'bottom_gray', 'bottom_brown',
                  'bottom_drawing', 'bottom_plain', 'bottom_plaid', 'bottom_horizontal-lines',
                  'bottom_vertical-lines']
        first_row= [0.0, 0.5, 0.5, 10.0, 12.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        sec_row= [1.0, 0.0, 0.0, 30.0, 12.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        third_row= [0.0, 0.5, 0.5, 10.0, 12.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        forth_row= [0.5, 0.5, 0.0, 30.0, 12.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        with open(train_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(first_row)
            writer.writerow(sec_row)
            writer.writerow(third_row)
            writer.writerow(forth_row)

        print(f"Created {train_file}")

    if not os.path.exists(wardrobe_file):
        # Create wardrobe file with header and first row
        header = ['Category', 'Color', 'Pattern', 'Length', 'Fit', 'ImagePath']
        first_row = ['overtop', 'null', 'null', 'null', 'null', '1711917857463.png']

        with open(wardrobe_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(first_row)
        print(f"Created {wardrobe_file}")

    return jsonify(message="Operation completed successfully"), 200

def get_rgb(color_name):
    try:
        # Fetch the hex code for the color name
        hex_color = mcolors.CSS4_COLORS[color_name]
        # Convert hex to RGB
        rgb = [float(int(hex_color[i:i + 2], 16)) / 100 for i in (1, 3, 5)]
    except KeyError:
        # Return a default RGB value if color name is not found
        rgb = [0.0, 0.0, 0.0]  # Assuming black as a default or you can choose another default
    return rgb


def open_wardrobe_data(user):

    df = pd.read_csv(f'wardrobe_{user}.csv', header=0)

    # COLORS
    color_dummies = pd.get_dummies(df['Color'], dtype=float)

    df = pd.concat([df, color_dummies], axis=1)

    df.drop('Color', axis=1, inplace=True)

    # PATTERN

    pattern_dummies = pd.get_dummies(df['Pattern'], dtype=float)

    df = pd.concat([df, pattern_dummies], axis=1)

    df.drop('Pattern', axis=1, inplace=True)

    # LENGTH
    length_mapping = {'short': 1, 'medium': 2, 'long': 3}

    df['Length'] = df['Length'].map(length_mapping)

    # FIT
    fit_mapping = {'tight': 1, 'fit': 2, 'loose': 3, 'oversize': 4}

    df['Fit'] = df['Fit'].map(fit_mapping)

    df.fillna(0, inplace=True)

    desired_order = ["Category", "Length", "Fit", "red", "orange", "yellow", "green", "blue", "purple", "black",
                     "white", "gray", "brown", "drawing", "plain", "plaid", "horizontal-lines",
                     "vertical-lines", "ImagePath"]
    df = df.reindex(columns=desired_order, fill_value=0)
    return df


def choose_index_with_probability(arr):
    # Define probabilities for each index
    if len(arr) == 4:
        probabilities = [60, 25, 10, 5]
    if len(arr) == 3:
        probabilities = [60, 25, 15]
    if len(arr) == 2:
        probabilities = [70, 30]
    if len(arr) == 1:
        probabilities = [100]

    rand_num = random.randint(0, 100)

    # Determine which index to return based on probabilities
    cumulative_prob = 0
    for i, prob in enumerate(probabilities):
        cumulative_prob += prob
        if rand_num <= cumulative_prob:
            return arr[i]


def nn_find_cloths(input, prediction, outfit_dataset, user, categories=["top", "overtop", "bottom"], flag=0):
    outfit = []
    concatenated_array = np.array(input)
    desired_order = ["Length", "Fit", "red", "orange", "yellow", "green", "blue", "purple", "black",
                     "white", "gray", "brown", "drawing", "plain", "plaid", "horizontal-lines",
                     "vertical-lines"]
    for category in categories:
        if category == "top":
            prediction_subset = prediction[:, 0:17]
        if category == "overtop":
            prediction_subset = prediction[:, 17:34]
        if category == "bottom":
            prediction_subset = prediction[:, 34:51]

        category_data = outfit_dataset[outfit_dataset["Category"] == category]

        nbrs = NearestNeighbors(algorithm='auto').fit(
            category_data.drop(["ImagePath", "Category"], axis=1))

        if category == "overtop":
            distances, indices = nbrs.kneighbors(prediction_subset, 1)
            most_similar_index = indices[0]
            print(most_similar_index)
            most_similar_index = choose_index_with_probability(most_similar_index)
            print(most_similar_index)
            most_similar_item = category_data.iloc[most_similar_index]
            if most_similar_item["ImagePath"] != no_overtop:
                try:
                    distances, indices = nbrs.kneighbors(prediction_subset, 3)
                except:
                    try:
                        distances, indices = nbrs.kneighbors(prediction_subset, 2)
                    except:
                        distances, indices = nbrs.kneighbors(prediction_subset, 1)

        else:
            try:
                distances, indices = nbrs.kneighbors(prediction_subset, 4)
            except:
                try:
                    distances, indices = nbrs.kneighbors(prediction_subset, 3)
                except:
                    try:
                        distances, indices = nbrs.kneighbors(prediction_subset, 2)
                    except:
                        distances, indices = nbrs.kneighbors(prediction_subset, 1)

        most_similar_index = indices[0]
        print(most_similar_index)
        most_similar_index = choose_index_with_probability(most_similar_index)
        print(most_similar_index)
        most_similar_item = category_data.iloc[most_similar_index]
        outfit.append(most_similar_item["ImagePath"])

        with open(f"current_{user}.csv", 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(np.append(np.array(most_similar_item), 1))

        concatenated_array = np.concatenate(
            (concatenated_array, category_data.reindex(columns=desired_order).iloc[most_similar_index]))
    return concatenated_array, outfit


def predict_outfit(condition, temperature, hour, user):
    input_vector = [condition, temperature, hour]
    # input_vector = [request.args.get('condition', 'sunny'), request.args.get('temperature', '20'), request.args.get('hour', '12')]
    # Load the training data from train.csv
    train_data = pd.read_csv(f'train_{user}.csv')

    # add
    input_columns = ["sunny", "cloudy", "rainy", "temp", "time"]

    X = train_data[input_columns]
    y = train_data.drop(columns=input_columns)

    # Train the XGBoost model
    model = xgb.XGBRegressor(objective='reg:squarederror')
    model.fit(X, y)
    if input_vector[0] == "sunny":
        input_vector = [1, 0, 0, input_vector[1], input_vector[2]]

    if input_vector[0] == "cloudy":
        input_vector = [0, 1, 0, input_vector[1], input_vector[2]]

    if input_vector[0] == "rainy":
        input_vector = [0, 0, 1, input_vector[1], input_vector[2]]

    new_input = pd.DataFrame([input_vector], columns=X.columns)
    prediction = model.predict(new_input)

    # finding item similar to predicting
    outfit_dataset = open_wardrobe_data(user)

    np.savetxt(f"current_{user}.csv", np.array(prediction), delimiter=',')

    concatenated_array, outfit = nn_find_cloths(input_vector, prediction, outfit_dataset,user)

    with open(f"train_{user}.csv", 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(concatenated_array)

    with open(f"current_images_{user}.csv", 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([outfit[1], outfit[0], outfit[2]])

    return [outfit[1], outfit[0], outfit[2]]


def delete_last_row_and_save(filename):
    df = pd.read_csv(filename)
    last_row = df.iloc[-1].values.tolist()  # Extract last row as list
    df = df.iloc[:-1]  # Remove the last row
    df.to_csv(filename, index=False)
    return last_row  # Return last row as list


def regenerate_outfit(user, flag=0):
    df = pd.read_csv(f'current_{user}.csv', header=None)
    df.loc[1:, flag_col] = 0
    df.to_csv(f'current_{user}.csv', index=False, header=False)

    current = pd.read_csv(f"current_{user}.csv", header=None)
    current_images = pd.read_csv(f"current_images_{user}.csv", header=None).values.tolist()[0]
    prediction = current.iloc[0].values.tolist()
    prediction = np.array([[float(item) for item in prediction]])
    outfit_dataset = open_wardrobe_data(user)

    input = delete_last_row_and_save(f"train_{user}.csv")[:5]

    concatenated_array, outfit = nn_find_cloths(input, prediction, outfit_dataset, user)

    with open(f"train_{user}.csv", 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(concatenated_array)

    with open(f"current_images_{user}.csv", 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([outfit[1], outfit[0], outfit[2]])

    if ([outfit[1], outfit[0], outfit[2]] == current_images or
        [outfit[1], outfit[0]] == [current_images[0], current_images[1]] or
        [outfit[0], outfit[2]] == [current_images[1], current_images[2]] or
        [outfit[2], outfit[1]] == [current_images[2], current_images[0]]) and flag <= 10:
        return regenerate_outfit(user, flag + 1)
    if [outfit[1], outfit[0], outfit[2]] == current_images and flag<=15:
        return regenerate_outfit(user, flag + 1)

    return [outfit[1], outfit[0], outfit[2]]


# make one for every kind of clothing
def change_top(user):
    current = pd.read_csv(f"current_{user}.csv", header=None)
    current_images = pd.read_csv(f"current_images_{user}.csv", header=None).values.tolist()[0]
    prediction = current.iloc[0].values.tolist()
    prediction = np.array([[float(item) for item in prediction]])
    current = current.iloc[1:]
    outfit_dataset = open_wardrobe_data(user)

    outfit_dataset = outfit_dataset[outfit_dataset['Category'] == "top"]

    # deleting items we offered already
    for index, row in current.iterrows():
        if row.values.tolist()[flag_col] == 1:
            outfit_dataset = outfit_dataset[outfit_dataset['ImagePath'] != row.values.tolist()[flag_col-1]]
    outfit_dataset = outfit_dataset[outfit_dataset['ImagePath'] != current_images[1]]
    if outfit_dataset.empty:
        df = pd.read_csv(f'current_{user}.csv', header=None)
        df.loc[1:, flag_col] = 0
        df.to_csv(f'current_{user}.csv', index=False, header=False)
        return change_top()

    last_pred = delete_last_row_and_save(f"train_{user}.csv")
    input = last_pred[:5]

    concatenated_array, outfit = nn_find_cloths(input, prediction, outfit_dataset, user, categories=["top"], flag=1)
    print(concatenated_array)
    last_pred[start_of_top+5:end_of_top+5] = concatenated_array[start_of_top+5:end_of_top+5]
    print(last_pred)

    with open(f"train_{user}.csv", 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(last_pred)

    with open(f"current_images_{user}.csv", 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([current_images[0], outfit[0], current_images[2]])

    return outfit[0]


def change_overtop(user):
    current = pd.read_csv(f"current_{user}.csv", header=None)
    current_images = pd.read_csv(f"current_images_{user}.csv", header=None).values.tolist()[0]
    print(current_images)
    prediction = current.iloc[0].values.tolist()
    prediction = np.array([[float(item) for item in prediction]])
    current = current.iloc[1:]
    outfit_dataset = open_wardrobe_data(user)
    outfit_dataset = outfit_dataset[outfit_dataset['Category'] == "overtop"]

    # deleting items we offered already
    for index, row in current.iterrows():
        if row.values.tolist()[flag_col] == 1:
            outfit_dataset = outfit_dataset[outfit_dataset['ImagePath'] != row.values.tolist()[flag_col-1]]
    outfit_dataset = outfit_dataset[outfit_dataset['ImagePath'] != current_images[0]]

    if outfit_dataset.empty:
        df = pd.read_csv(f'current_{user}.csv', header=None)
        df.loc[1:, flag_col] = 0
        df.to_csv(f'current_{user}.csv', index=False, header=False)
        return change_overtop()

    last_pred = delete_last_row_and_save(f"train_{user}.csv")
    input = last_pred[:5]

    concatenated_array, outfit = nn_find_cloths(input, prediction, outfit_dataset, user, categories=["overtop"], flag=1)
    print(concatenated_array)
    last_pred[start_of_overtop+5:end_of_overtop+5] = concatenated_array[start_of_top+5:end_of_top+5]
    print(last_pred)

    with open(f"train_{user}.csv", 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(last_pred)

    with open(f"current_images_{user}.csv", 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([outfit[0], current_images[1], current_images[2]])

    return outfit[0]


def change_bottom(user):
    current = pd.read_csv(f"current_{user}.csv", header=None)
    current_images = pd.read_csv(f"current_images_{user}.csv", header=None).values.tolist()[0]
    print(current_images)
    prediction = current.iloc[0].values.tolist()
    prediction = np.array([[float(item) for item in prediction]])
    current = current.iloc[1:]
    outfit_dataset = open_wardrobe_data(user)

    outfit_dataset = outfit_dataset[outfit_dataset['Category'] == "bottom"]
    # deleting items we offered already
    for index, row in current.iterrows():
        if row.values.tolist()[flag_col] == 1:
            outfit_dataset = outfit_dataset[outfit_dataset['ImagePath'] != row.values.tolist()[flag_col-1]]
    outfit_dataset = outfit_dataset[outfit_dataset['ImagePath'] != current_images[2]]

    if outfit_dataset.empty:
        df = pd.read_csv(f'current_{user}.csv', header=None)
        df.loc[1:, flag_col] = 0
        df.to_csv(f'current_{user}.csv', index=False, header=False)
        return change_bottom()

    last_pred = delete_last_row_and_save(f"train_{user}.csv")
    input = last_pred[:5]

    concatenated_array, outfit = nn_find_cloths(input, prediction, outfit_dataset, user, categories=["bottom"], flag=1)
    print(concatenated_array)
    last_pred[start_of_bottom+5:] = concatenated_array[start_of_top+5:end_of_top+5]
    print(last_pred)

    with open(f"train_{user}.csv", 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(last_pred)

    with open(f"current_images_{user}.csv", 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([current_images[0], current_images[1], outfit[0]])

    return outfit[0]



if __name__ == "__main__":
    app.run(debug=True, port=5001)
