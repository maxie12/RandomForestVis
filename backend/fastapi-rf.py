from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from fastapi.responses import ORJSONResponse
from fastapi.middleware.gzip import GZipMiddleware

import pandas as pd
import numpy as np 
import os
#import io
import joblib
import json
import csv
import multiprocessing as mp

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import _tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
import umap.umap_ as umap

#from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from dynamicTreeCut import cutreeHybrid

import copy
import time
import logging

import orjson
import mmap

# Configure Time logging
logging.basicConfig(filename='time_measurements.log', level=logging.INFO, format='%(asctime)s - %(message)s')

app = FastAPI(default_response_class=ORJSONResponse)

origins = ["http://127.0.0.1:3030", "http://127.0.0.1:8080", "http://localhost:3030", "http://localhost:8080"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"]
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# serves static files
app.mount("/static", StaticFiles(directory="static"), name="static")


### Static Files ####
@app.get("/")
def root():
    return FileResponse('../dist/index.html')


@app.get("/{file_path:path}")
async def serve_dist_files(file_path: str):
    file_location = f"../dist/{file_path}"
    
    # Check if the requested file exists
    if not os.path.exists(file_location):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_location)    

# get classification, missclassication for each tree, also counts for each node the type of data samples that goes through it
def get_tree_classification_data(tree, i, X, y):
    counts = get_data_in_tree(tree, X, y)

    class_names = np.unique(y).tolist()
    class_lookup = {}
    # numerical class names can differ from their index e.g. glass dataset 6 classes but one is called 7. 
    # so, we work with the index of a class
    for idx, class_name in enumerate(class_names):
        class_lookup[class_name] = idx

    tree_terminal_nodes = tree.apply(X)
    predictions = tree.predict(X)
    y = [class_lookup[int(true_label)] for true_label in y]
    tree_accuracy = accuracy_score(y, predictions)

    # Initialize dictionaries to count samples in each leaf
    leaf_node_counts = {}
    correct_predictions_counts = {}
    wrong_predictions_counts = {}
    for node, prediction, true_label in zip(tree_terminal_nodes, predictions, y):
        true_label = int(true_label)
        pred = int(prediction)   
        if node not in leaf_node_counts:
            leaf_node_counts[node] = 0
            correct_predictions_counts[node] = 0
            wrong_predictions_counts[node] = {"count": 0}
    
        leaf_node_counts[node] += 1
        if pred == true_label:
            correct_predictions_counts[node] += 1 
        else:
            if true_label not in wrong_predictions_counts[node]:
                wrong_predictions_counts[node][true_label] = {"count": 0}
            
            wrong_predictions_counts[node]["count"] += 1
            wrong_predictions_counts[node][true_label]["count"] += 1
            if prediction not in wrong_predictions_counts[node][true_label]:
                wrong_predictions_counts[node][true_label][pred] = 0
            wrong_predictions_counts[node][true_label][pred] += 1    

    classification = {"values": counts, "accuracy": tree_accuracy, "correct_predictions_counts": correct_predictions_counts, "wrong_predictions_counts": wrong_predictions_counts}

    return classification

# counts for each node the type of data samples that goes through it
def get_data_in_tree(clf, X, y):
    # Get the node indicator (i.e., the decision path) for the test samples
    node_indicator = clf.decision_path(X)

    # Initialize a counter array with shape (number of nodes, number of classes)
    num_nodes = clf.tree_.node_count
    class_names = np.unique(y).tolist()
    num_classes = len(class_names)
    counts = np.zeros((num_nodes, num_classes))
    class_lookup = {}
    # numerical class names can differ from their index e.g. glass dataset 6 classes but one is called 7. 
    # so, we work with the index of a class
    for idx, class_name in enumerate(class_names):
        class_lookup[class_name] = idx

    # Increment the count for each class at each node
    for sample_id in range(len(X)):
        node_index = node_indicator.indices[node_indicator.indptr[sample_id]:node_indicator.indptr[sample_id + 1]]
        for node_id in node_index:
            label = class_lookup[y[sample_id]]
            counts[node_id, label] += 1

    return counts

# get the tree data for a node-link vis in the frontend
def get_node_link_tree(tree, classification, class_names, feature_names=None):
    # Get a node-link representation of the decision tree.
    tree_ = tree.tree_
    feature_names = (
        feature_names if feature_names is not None else [f"feature_{i}" for i in tree_.feature]
    )
    node_count = tree_.node_count
    children_left = tree_.children_left
    children_right = tree_.children_right
    features = tree_.feature
    thresholds = tree_.threshold
    values = classification["values"] # this are test samples / and now also training
    tree_values = tree_.value  #this are only training samples.

    def update_ancestors(ancestors, orientation):
        feature = ancestors["feature"]
        threshold = ancestors["threshold"]
        ancestors = ancestors["ancestors"]
        if feature not in ancestors:
            ancestors[feature] = [] #[threshold]
        ancestors[feature].append({"threshold": threshold, "orientation": orientation})

        return ancestors


    # Recursive function to extract node data
    def recurse(node_id, orientation, ancestors):
        updated_ancestors = ancestors["ancestors"]
        if orientation != "root":
            updated_ancestors = update_ancestors(ancestors, orientation)       
        node = {
            "node_id": int(node_id),
            "feature": feature_names[features[node_id]],
            "threshold": thresholds[node_id],
            "values": values[node_id].tolist(), # only for training data values[node_id][0].tolist(),
            "children": [],
            "orientation": orientation,
            "ancestorFeatures": {k: v[:] for k, v in updated_ancestors.items()}
        }
        tempAncestors = {"ancestors": updated_ancestors, "feature": feature_names[features[node_id]], "threshold": thresholds[node_id] }
        
        #save accuracy in root    
        if node_id == 0:
            node["accuracy"] = classification["accuracy"]

        # Check if the current node is a leaf node
        if children_left[node_id] == _tree.TREE_LEAF and children_right[node_id] == _tree.TREE_LEAF:
            # in case no datapoint ends up in this leaf
            correct_predictions = 0
            wrong_predictions = {"count": 0}
            # Include prediction value for leaf nodes
            node["feature"] = "Prediction"
            node["threshold"] = class_names[tree_values[node_id].argmax()]  # Assuming values are class probabilities

            if node_id in classification["correct_predictions_counts"]:
                correct_predictions = classification["correct_predictions_counts"][node_id]
            
            if node_id in classification["wrong_predictions_counts"]:
                wrong_predictions = classification["wrong_predictions_counts"][node_id]

            node["correctPredictions"] = correct_predictions
            node["wrongPredictions"] = wrong_predictions


        if children_left[node_id] != _tree.TREE_LEAF:
            left_child = recurse(children_left[node_id], "left", copy.deepcopy(tempAncestors))
            node["children"].append(left_child)
        else:
            left_child = None

        if children_right[node_id] != _tree.TREE_LEAF:
            right_child = recurse(children_right[node_id], "right", copy.deepcopy(tempAncestors))
            node["children"].append(right_child)
        else:
            right_child = None

        # Aggregate the correct and wrong predictions from children
        if left_child and right_child:
            node["correctPredictions"] = left_child["correctPredictions"] + right_child["correctPredictions"]
            node["wrongPredictions"] = aggregateWrongPredictions(copy.deepcopy(left_child["wrongPredictions"]), copy.deepcopy(right_child["wrongPredictions"]))
            node["correctPredictionsClassesCounts"] = aggregateCorrectPredictions(left_child, right_child)
        elif left_child:
            node["correctPredictions"] = left_child["correctPredictions"]
            node["wrongPredictions"] = left_child["wrongPredictions"]
            node["correctPredictionsClassesCounts"] = {left_child["threshold"]: left_child["correctPredictions"]}
        elif right_child:
            node["correctPredictions"] = right_child["correctPredictions"]
            node["wrongPredictions"] = right_child["wrongPredictions"]
            node["correctPredictionsClassesCounts"] = {right_child["threshold"]: right_child["correctPredictions"]}

        return node

    
    # Start from the root
    tree_data = recurse(0, "root", {"ancestors": {}})
    return tree_data

def aggregateCorrectPredictions(left_child, right_child):
    combined_counts = {}
    # leaf case
    if "correctPredictionsClassesCounts" not in left_child:
        combined_counts[left_child["threshold"]] = left_child["correctPredictions"]
    else:    
        # Aggregate counts from the left child
        for class_name, count in left_child["correctPredictionsClassesCounts"].items():
            if class_name in combined_counts:
                combined_counts[class_name] += count
            else:
                combined_counts[class_name] = count

    # leaf case
    if "correctPredictionsClassesCounts" not in right_child:
        if right_child["threshold"] not in combined_counts:
            combined_counts[right_child["threshold"]] = right_child["correctPredictions"]
        else:
            combined_counts[right_child["threshold"]] += right_child["correctPredictions"]
    else: 
        # Aggregate counts from the right child
        for class_name, count in right_child["correctPredictionsClassesCounts"].items():
            if class_name in combined_counts:
                combined_counts[class_name] += count
            else:
                combined_counts[class_name] = count

    return combined_counts

def aggregateWrongPredictions(prediction1, prediction2):
    prediction1["count"] += prediction2["count"]
    for original_class in prediction2:
        if original_class != 'count':
            if not original_class in prediction1:
                prediction1[original_class] = prediction2[original_class]
            else:
                #sum
                for misclassified_as in prediction2[original_class]:
                    if misclassified_as != 'count':
                        if not misclassified_as in prediction1[original_class]:
                            prediction1[original_class][misclassified_as] = prediction2[original_class][misclassified_as]
                        else:
                            prediction1[original_class][misclassified_as] += prediction2[original_class][misclassified_as]
                prediction1[original_class]["count"] += prediction2[original_class]["count"]
    return prediction1

### Routes ###
# upload own data and check if data allready exist
# last column is always class
@app.post("/upload_data")
async def upload_data(file: UploadFile = File(...)):
    start_time = time.time() 
    dataset_name = file.filename.replace(".csv", "")
    filename = "static/datasets/" + file.filename
    # in case we do not save the file
    #content = await file.read()
    #df = pd.read_csv(io.StringIO(content.decode('utf-8')))

    # Save the uploaded file
    with open(filename, "wb") as buffer:
        buffer.write(await file.read())

    X, y, feature_names, class_names, class_counts, feature_values = get_dataset(filename)
    rules_data, cluster_data, tree_data, classification_data, depth_data = check_for_files(dataset_name, class_names, feature_names, X, y)

    projection = compute_projection(dataset_name, "MDS")

    end_time = time.time() 
    print(end_time - start_time)
    return {"cluster": cluster_data, "rules": rules_data, "depth_data": depth_data, "projection": projection, 
    "classes": class_counts, "features": feature_values, "trees": tree_data, "classification": classification_data}


# get pre-computed data (test dataset) -> currently, we only use this for the start page with iris 
@app.post("/get_data")
async def get_data(request: Request):
    body = await request.body()
    # Convert bytes to string
    dataset_name = body.decode()

    filename = "static/datasets/" + dataset_name + ".csv"
    _, _, feature_names, class_names, class_counts, feature_values = get_dataset(filename)

    rules_data, cluster_data, tree_data, classification_data, depth_data = check_for_files(dataset_name, class_names, feature_names)

    proj_name = "MDS"
    projection = compute_projection(dataset_name, proj_name)
    return {"cluster": cluster_data, "rules": rules_data, "depth_data": depth_data, "projection": projection, 
    "classes": class_counts, "features": feature_values, "trees": tree_data, "classification": classification_data}


### Helper functions

# temp function
def load_iris_dataset():
    filename = "static/datasets/Iris.csv"
    X, y, _, _, _, _ = get_dataset(filename)

    return X, y

# use orjson and mmap for speed up
def read_json(path):
    with open(path, "r") as f:
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as m:
            data = orjson.loads(m.read().decode('utf-8'))
    return data

def read_trees(path):
    vectorData = []
    with open(path, 'rb') as f:
        distances = np.load(f)
    for x in range(len(distances)):
        vectorData.append(list(distances[x])) 

    return vectorData, distances

def read_projection(path):
    with open(path, 'rb') as f:
        projection = np.load(f)
    
    return projection 

# Define custom orders
custom_orders = {
    'buying price': ['low', 'med', 'high', 'vhigh'],
    'maintenance cost': ['low', 'med', 'high', 'vhigh'],
    'lug_boot': ['small', 'med', 'big'],
    'safety': ['low', 'med', 'high']
}

def data_imputation(df, categorical_columns, numerical_columns):
    # Impute missing values in categorical columns with "unknown"
    for col in categorical_columns:
        df[col] = df[col].fillna('unknown')

    # Compute the median for each numerical column and use it for imputation
    for col in numerical_columns: 
        df[col] = df[col].fillna(df[col].median(skipna=True))

    return df

def get_dataset(path):
    start_time = time.time()    
    df = pd.read_csv(path, sep=r",|;|\t|\|", engine='python')
    # Identify categorical columns i.e. strings
    categorical_columns = df.select_dtypes(include=['object']).columns
    numerical_columns = df.select_dtypes(exclude=['object']).columns
    # Perform data imputation - in case of missing values
    df = data_imputation(df, categorical_columns, numerical_columns)
    # Assume the last column is the class
    y_label = df.columns[-1]    
    # use orginal class names and not numbers
    class_names = df[y_label].unique().tolist()
    # Create class_counts dictionary in the same order as class_names
    value_counts = df[y_label].value_counts().to_dict()
    class_counts = {
        class_name: {"count": value_counts[class_name], "index": idx}
        for idx, class_name in enumerate(class_names)
    }
    # Create a dictionary to hold metadata about categorical features
    categorical_feature_info = {}
    # Convert categorical columns to numerical 
    for col in categorical_columns:
        categories = sorted(df[col].unique())
        if col in custom_orders:
            # Custom order for specific column
            categories = custom_orders[col]
        
        df[col] = pd.Categorical(df[col], categories=categories, ordered=True).codes
        categories_norminal = pd.Categorical(categories, categories=categories, ordered=True).codes
        # Store category names for each categorical feature
        categorical_feature_info[col] = {
            'is_categorical': True,
            'categories': categories,
            'categories_norminal': categories_norminal.tolist(),
        }

    # Handle numerical columns: Store min and max values
    numerical_feature_info = {}
    for col in numerical_columns:
        min_val = df[col].min()
        max_val = df[col].max()
        numerical_feature_info[col] = {
            'is_categorical': False,
            'range': (min_val, max_val)
        }

    y = df[y_label]
    # drop target class
    X = df.drop(columns=[y_label])
    feature_names = list(X.columns.values)
    # Normalize the distribution for each feature between 0 and 1
    scaler = MinMaxScaler()
    df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])
    
    # Save the scaled values for each feature with the info object
    feature_infos = {}
    for i, feature_name in enumerate(feature_names):
        scaled_values = df.iloc[:, i].tolist()
        feature_info = numerical_feature_info[feature_name] if feature_name in numerical_feature_info else categorical_feature_info[feature_name]
        feature_info["scaled_values"] = scaled_values
        feature_info["numerical_categories"] = sorted(set(scaled_values)) if feature_info["is_categorical"] else []
        feature_infos[feature_name] = feature_info
        feature_info["index"] = i

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Get Dataset, Elapsed time: {elapsed_time} seconds")   
    return X, y, feature_names, class_names, class_counts, feature_infos

# convert tuple with most NumPy values to native Python types. for json serialization
def convert_tuple(val):
    return (val[0].item(), val[1].item())

def get_rf_model(rf_path, X_train, y_train):
    if os.path.exists(rf_path):
        rf_model = joblib.load(rf_path)

    else:
        rf_model = compute_rf_model(rf_path, X_train, y_train)
    return rf_model    

def compute_rf_model(rf_path, X_train, y_train):
    start_time = time.time()
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42) 
    # Train the model on the training set
    rf_model.fit(X_train, y_train)
    # save model
    joblib.dump(rf_model, rf_path)
    
    # End measuring time
    end_time = time.time()
    # Calculate elapsed time
    elapsed_time = end_time - start_time
    # Log elapsed time
    logging.info(f"RF Model {rf_path}, Elapsed time: {elapsed_time} seconds")    
    return rf_model

# compute rules and clustering if dataset is new
def check_for_files(dataset_name, class_names, feature_names, X=None, y=None):

    cluster_path = 'static/clustering/complete_linkage_' + dataset_name + '_rule_interval.json'  
    rules_path = "static/rules/Rule_Data_" + dataset_name + ".json"
    trees_path = "static/trees/trees_" + dataset_name + "_rule_interval.npy"
    rf_path = "static/models/random_forest_" + dataset_name +".joblib"
    node_link_path = "static/node_link/node_link_trees_" + dataset_name + ".json"
    classification_path = "static/classification/classification_" + dataset_name + ".json"

    rules_data = []
    cluster_data = []
    tree_data = []
    classification = []
    tree_classification = []
    depth_data = []
    
    cluster_exists = os.path.exists(cluster_path)
    trees_exists = os.path.exists(trees_path)
    rules_exists = os.path.exists(rules_path)
    model_exists = os.path.exists(rf_path)
    node_link_exists = os.path.exists(node_link_path)
    classification_exists = os.path.exists(classification_path)

    # Temp: data was not uploaded -> use Iris
    if X is None or y is None:
        X, y = load_iris_dataset()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
    rf_model = get_rf_model(rf_path, X_train, y_train)
    clfs = rf_model.estimators_
    num_features = rf_model.n_features_in_

    # remove header to skip warnings
    X_values = X.values
    y_values = y.values
    X_test = X_test.values
    y_test = y_test.values

    # Extract the node-link structure of the decision tree
    if node_link_exists:
        tree_data = read_json(node_link_path)
    else:
        start_time = time.time()
        tree_classification = [get_tree_classification_data(estimator, i, X_values, y_values) for i, estimator in enumerate(clfs)]
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f"Get Tree Classification Data, Elapsed time: {elapsed_time} seconds")   
        start_time = time.time()
        tree_data = [get_node_link_tree(estimator, tree_classification[i], class_names, feature_names) for i, estimator in enumerate(clfs)]  
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f"Get Node Link Data, Elapsed time: {elapsed_time} seconds")   
        with open(node_link_path, 'w') as json_file:
            json.dump(tree_data, json_file)
    
    if classification_exists:
        classification = read_json(classification_path)
    else:
        # use test data for overall classification
        classification = get_classification(rf_model, X_test, y_test)
        with open(classification_path, 'w') as json_file:
            json.dump(classification, json_file)

    # rules exist load rules
    if rules_exists:
        rules_data = read_json(rules_path)

    # no rules -> compute them
    else:   
        vectorData, distances, rules_data = compute_distances_and_extract_rules(trees_path, rules_path, clfs, num_features, class_names, X, tree_classification)

    trees_exists = os.path.exists(trees_path)

    # clustering allready exists
    if cluster_exists:
        cluster_data = read_json(cluster_path)
    # clustering does not exist, but distances exist
    elif trees_exists:
        vectorData, distances =  read_trees(trees_path)
        cluster_data = dynamic_multilevel_cut(cluster_path, vectorData, dataset_name, clfs)

    depth_data = compute_depth_plot(dataset_name, tree_classification)

    return rules_data, cluster_data, tree_data, classification, depth_data

def count_correct_wrong_classification_ensemble(prediction, groundtruth):
    correct_counts = {}
    wrong_counts = {}

    class_names = np.unique(np.unique(groundtruth).tolist() + np.unique(prediction).tolist())
    class_lookup = {}
    # numerical class names can differ from their index e.g. glass dataset 6 classes but one is called 7. 
    # so, we work with the index of a class
    for idx, class_name in enumerate(class_names):
        class_lookup[class_name] = idx
    for i in range(len(prediction)):
        pred = int(prediction[i])
        gt = int(groundtruth[i])
        index_pred = class_lookup[pred]
        index_gt = class_lookup[gt] 
        if index_pred not in correct_counts:
            correct_counts[index_pred] = 0
            wrong_counts["count"] = 0
            wrong_counts[index_pred] = {"count": 0}

        if gt != pred:
            # save for each wrong classification 
            wrong_counts["count"] += 1
            wrong_counts[index_pred]["count"] += 1
            if index_gt not in wrong_counts[index_pred]:
                wrong_counts[index_pred][index_gt] = 0
            wrong_counts[index_pred][index_gt] += 1
        else:
            correct_counts[index_gt] += 1

    return correct_counts, wrong_counts

def get_classification(rf_model, X, y):
    start_time = time.time()
    y_pred = rf_model.predict(X)
    # Evaluate the model's accuracy
    accuracy = accuracy_score(y, y_pred)
    # get probabilities for each data sample -> agreement on trees
    y_pred_proba = rf_model.predict_proba(X)
    correct_predictions_model_counts, wrong_predictions_model_counts = count_correct_wrong_classification_ensemble(y_pred, y)
    model_classification = {"agreement": y_pred_proba.tolist(), "correctPredictions": correct_predictions_model_counts, "wrongPredictions": wrong_predictions_model_counts, "accuracy": accuracy}

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Get Model Classification Data, Elapsed time: {elapsed_time} seconds")           
    return model_classification


### Compute Data for Depth Plot ###
def compute_depth_plot(dataset_name, tree_classification):
    start_time = time.time()

    #load rf model
    rf_path = "static/models/random_forest_" + dataset_name + ".joblib"
    model_exists = os.path.exists(rf_path)

    data_path =  "static/datasets/" + dataset_name + ".csv"
    data_exists = os.path.exists(data_path)
    depth_path = "static/Depth_data/RF_" + dataset_name + ".csv"

    depth_data = [] 
    # load data if exist
    if os.path.exists(depth_path):
        # uses , as seperator
        with open(depth_path, newline='\n') as csvfile:
            csvreader = csv.DictReader(csvfile)
            for row in csvreader:
                depth_data.append(row)
        return depth_data

    # no dataset
    if not data_exists:
        return []

    X, y, features, class_names, _, _ = get_dataset(data_path)

    X_train = []
    y_train = []
    if not model_exists:
        # in case we need to create the rf_model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = get_rf_model(rf_path, X_train, y_train)

    def accumulate_predictions(node_id, values):
        if is_leaves[node_id]:
            predicted_class = values[node_id].argmax()
            correct_predictions = tree_classification[i]["correct_predictions_counts"].get(node_id, 0)
            correct_predictions_classes_counts = {predicted_class: correct_predictions}
            wrong_predictions = tree_classification[i]["wrong_predictions_counts"].get(node_id, {"count": 0})
        else:
            left_child = children_left[node_id]
            right_child = children_right[node_id]

            left_predictions = accumulate_predictions(left_child, values)
            right_predictions = accumulate_predictions(right_child, values)

            correct_predictions = left_predictions["correctPredictions"] + right_predictions["correctPredictions"]
            correct_predictions_classes_counts = {}

            for class_name, count in left_predictions["correctPredictionsClassesCounts"].items():
                if class_name not in correct_predictions_classes_counts:
                    correct_predictions_classes_counts[class_name] = 0
                correct_predictions_classes_counts[class_name] += count

            for class_name, count in right_predictions["correctPredictionsClassesCounts"].items():
                if class_name not in correct_predictions_classes_counts:
                    correct_predictions_classes_counts[class_name] = 0
                correct_predictions_classes_counts[class_name] += count

            wrong_predictions = {"count": left_predictions["wrongPredictions"]["count"] + right_predictions["wrongPredictions"]["count"]}

            for original_class in left_predictions["wrongPredictions"]:
                if original_class != "count":
                    if original_class not in wrong_predictions:
                        wrong_predictions[original_class] = {}
                    for misclassified_as, count in left_predictions["wrongPredictions"][original_class].items():
                        if misclassified_as != "count":
                            if misclassified_as not in wrong_predictions[original_class]:
                                wrong_predictions[original_class][misclassified_as] = 0
                            wrong_predictions[original_class][misclassified_as] += count

            for original_class in right_predictions["wrongPredictions"]:
                if original_class != "count":
                    if original_class not in wrong_predictions:
                        wrong_predictions[original_class] = {}
                    for misclassified_as, count in right_predictions["wrongPredictions"][original_class].items():
                        if misclassified_as != "count":
                            if misclassified_as not in wrong_predictions[original_class]:
                                wrong_predictions[original_class][misclassified_as] = 0
                            wrong_predictions[original_class][misclassified_as] += count

        predictions[node_id] = {
            "correctPredictions": correct_predictions,
            "correctPredictionsClassesCounts": correct_predictions_classes_counts,
            "wrongPredictions": wrong_predictions
        }
        return predictions[node_id]


    f = open(depth_path, "w")
    #f.write("tree_id,node_id,is_leave,samples,impurity,split_feature,threshold,min_feature,max_feature,children\n")
    f.write("tree_id,node_id,is_leave,split_feature,threshold,children,correctPredictions,correctPredictionsClassesCounts,wrongPredictions\n")
    for i in range(len(rf_model.estimators_)):
        model = rf_model.estimators_[i]
        
        DTree = model.tree_
        children_left = DTree.children_right
        children_right = DTree.children_left
        n_nodes = DTree.node_count
        values = DTree.value # only training samples 
        classification_values = tree_classification[i]["values"]

        feature_name = [
            features[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in DTree.feature
        ]

        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves  = np.zeros(shape=n_nodes, dtype=bool)
        stack  = [(0, -1, -1)] #seed is the root node id and its parent depth and parent id
        parent = {}
        nodes = [0]
        edges = {}
        predictions = {}
        #samples = {}
        #impurities = {}
        feature_and_threshold = {}
        while len(stack) > 0:
            node_id, parent_depth, parent_node_id = stack.pop()
            node_depth[node_id] = parent_depth + 1

            if (node_id != 0):
                # Check if key already exists in dict
                childs = []
                if parent_node_id in edges:
                    # get the list of children
                    childs = edges[parent_node_id]
                childs.append(node_id)
                edges[parent_node_id] = childs
                
                parent[node_id] = parent_node_id
                nodes.append(node_id)
                
                if parent_node_id not in feature_and_threshold:
                    feature = feature_name[parent_node_id]
                    threshold = DTree.threshold[parent_node_id]
                    threshold = "{:.2f}".format(threshold)
                    feature_and_threshold[parent_node_id] = (feature, threshold)
                    
            #samples[node_id]    = DTree.n_node_samples[node_id]
            #impurities[node_id] = "{:.3f}".format(DTree.impurity[node_id])
            
            if (children_right[node_id] != children_left[node_id]):
                stack.append((children_left[node_id], parent_depth + 1, node_id))
                stack.append((children_right[node_id], parent_depth + 1, node_id))
            else:
                is_leaves[node_id] = True
                predictions[node_id] = {}

                correct_predictions = tree_classification[i]["correct_predictions_counts"].get(node_id, 0)
                predicted_class = classification_values[node_id].argmax()
                correct_predictions_classes_counts = {predicted_class: correct_predictions}
                
                wrong_predictions = tree_classification[i]["wrong_predictions_counts"].get(node_id, {"count": 0})

                predictions[node_id]["correctPredictions"] = correct_predictions
                predictions[node_id]["correctPredictionsClassesCounts"] = correct_predictions_classes_counts
                predictions[node_id]["wrongPredictions"] = wrong_predictions

        for node in nodes:
            prediction = accumulate_predictions(node, values)
            #line = str(i) + "," + str(node) + "," + str(is_leaves[node]) + "," + str(samples[node]) + "," + str(impurities[node])
            line = str(i) + "," + str(node) + "," + str(is_leaves[node]) 
            
            if not is_leaves[node]:
                line += "," + "\"" + feature_and_threshold[node][0] + "\""
                line += "," + feature_and_threshold[node][1]
                
                #line += "," + str(X[feature_and_threshold[node][0]].min())
                #line += "," + str(X[feature_and_threshold[node][0]].max())
                
                # get children
                line += ",\""
                for child in edges[node]:
                    line += str(child) + ";"
                line = line[:-1]
                line += "\""
                
            else:
                line += ",\"\",\"\",\"\""
            
            # save predictions 
            line += "," + str(prediction["correctPredictions"])         
            correct_predictions_classes_counts = prediction["correctPredictionsClassesCounts"]
            correct_predictions_classes_counts_str = ""
            for class_name, count in correct_predictions_classes_counts.items():
                correct_predictions_classes_counts_str += f"{class_name};{count};"

            if correct_predictions_classes_counts_str.endswith(";"):
                correct_predictions_classes_counts_str = correct_predictions_classes_counts_str[:-1]

            line += ",\"" + correct_predictions_classes_counts_str + "\""


            #TotalCount;Original Class;MisclassifiedAs;Count;...;MisclassifiedAs;Count
            line += ",\"" + str(prediction["wrongPredictions"]["count"])
            for original_class in prediction["wrongPredictions"]:
                if original_class != 'count':
                    line += ";" + str(original_class)
                    for misclassified_as in prediction["wrongPredictions"][original_class]:
                        if misclassified_as != 'count':
                            line += ";" + str(misclassified_as) + ";" + str(prediction["wrongPredictions"][original_class][misclassified_as])

            line += "\""
            f.write(line)
            
            if not ((i == len(rf_model.estimators_)-1) and (node == nodes[-1])):
                f.write('\n')
    f.close()
    
    with open(depth_path, newline='\n') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            depth_data.append(row)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Depth Plot {depth_path}, Elapsed time: {elapsed_time} seconds")   


    return depth_data


### Dimensional Reduction ### 

def compute_projection(dataset_name, proj_name):
    start_time = time.time()
    trees_path = "static/trees/trees_" + dataset_name + "_rule_interval.npy"
    proj_path = "static/projections/" + proj_name + "_" + dataset_name + "_projection.npy"
    trees_exists = os.path.exists(trees_path)
    projection_exists = os.path.exists(proj_path)
    proj = []
    # check if projection exists
    if projection_exists:
        proj = read_projection(proj_path)
    # we assume that the trees are allready computed
    elif trees_exists:
        vectorData, distances = read_trees(trees_path)
        if proj_name == "MDS":
            proj = compute_mds(vectorData)

        if proj_name == "PCA":
            proj = compute_pca(vectorData)

        if proj_name == "UMAP":
            proj = compute_umap(vectorData)

        if proj_name == "TSNE":
            proj = compute_tsne(vectorData)

        # save projection 
        np.save(proj_path, proj, allow_pickle=True)

    if isinstance(proj, np.ndarray):
        proj = proj.tolist()

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Projection {proj_path}, Elapsed time: {elapsed_time} seconds")   
    return proj

def compute_mds(vectorData):
    mds = MDS(n_components=2, random_state=42)
    u = mds.fit_transform(vectorData)

    return u   

def compute_pca(vectorData):
    pca = PCA(n_components=2)
    u = pca.fit_transform(vectorData)

    return u  

def compute_tsne(vectorData):
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    u = tsne.fit_transform(vectorData)

    return u       

def compute_umap(vectorData):
    fit = umap.UMAP(n_components=2,random_state=42, n_neighbors=5)
    u = fit.fit_transform(vectorData)

    return u

### Clustering ###
# hierarchical clustering with complete linkage, followed by a mutlilevelcut with cutreeHybrid
def dynamic_multilevel_cut(cluster_path, vectorData, dataset, trees):
    start_time = time.time() 
    # creates a condensed distance matrix -> looks worse
    # vectorData = squareform(vectorData)

    # Perform hierarchical clustering with complete linkage
    linkage_matrix = linkage(vectorData, method='complete', optimal_ordering=True)
    
    # decide cluster numbers in 0.1 distance steps. from min to max value
    """min_distance = np.around(np.min(linkage_matrix[:, 2]), decimals=1)
    max_distance = np.around(np.max(linkage_matrix[:, 2]), decimals=1)
    steps = [min_distance]
    current_distance = min_distance
    while current_distance < max_distance:
        current_distance += 0.1
        steps.append(np.around(current_distance, decimals=1))
    """
    
    # cutHeight: Maximum joining heights that will be considered. 
    # It defaults to 99of the range between the 5th percentile and the maximum of the joining heights on the dendrogram.
    # default value minClusterSize = 5
    clusterings = []
    sizeLookup = {}
    maxClusterLookup = {}
    stepLookup = {}
    elbow_y_values = []
    # cutreeHybrid als Default und dann fCluster mit Clustergröße oder Threshold
    for i in range(1, 21):
        cuttree = cutreeHybrid(linkage_matrix, np.array(vectorData), minClusterSize = i, deepSplit = 1)
        labels = cuttree["labels"]
        n_cluster = int(np.max(labels))
        centroids = selectRepresentativeTree(vectorData, labels, n_cluster)
        #classification = {}
        # for each cluster number i go through the labels
        """for c in range(1, n_cluster + 1):
            ensemble = []
            #do a majority voting with all trees in a cluster
            for j in range(len(labels)):
                if labels[j] == c:
                    # get prediction of this tree 
                    tree = trees[c]
                    ensemble.append(tree)
            # based on scikit-learn Voting Classifier with soft voting    
            avg = np.average(np.asarray([clf.predict_proba(X) for clf in ensemble]), axis=0)
            voting = np.argmax(avg, axis=1)
            # compare voting with ground truth
            correct_predictions, wrong_predictions = count_correct_wrong_classification_ensemble(voting, y)
            
            accuracy = accuracy_score(y, voting)
            classification[c] = {"accuracy": accuracy, "correctPredictions": correct_predictions, "wrongPredictions": wrong_predictions}
       """

        # Traverse the linkage matrix as a tree
        # linkage is used for the clustering but we do not use the json format -> could be used for interactive cuts
        #data = create_linkage_json(labels, linkage_matrix) 
        #sizeLookup[i] = {"classification": classification, "linkage_data": data, "labels": labels.tolist(), "n_cluster": n_cluster, "centroids": centroids}
        sizeLookup[i] = {"labels": labels.tolist(), "n_cluster": n_cluster, "centroids": centroids}
        elbow_y_values.append(n_cluster)

    # for elbow method
    max_slope_change_index = find_elbow_point_after_max_change(elbow_y_values)

    """for i in range(2, 100):
        labels_max_clust = create_cluster_json(linkage_matrix, i, 'maxclust')
        maxClusterLookup[i] = {"labels": labels_max_clust.tolist(), "n_cluster": int(np.max(labels_max_clust))}

    for i in range(len(steps)):
        labels_step = create_cluster_json(linkage_matrix, steps[i], 'distance')
        stepLookup[steps[i]] = {"labels": labels_step.tolist(), "n_cluster": int(np.max(labels_step))}
    """

    clusters = {"minClusterSize": sizeLookup, "elbow": max_slope_change_index} #, "maxclust": maxClusterLookup, "distance": stepLookup}


    # go through each cluster majority votgin based on probability of the single trees
    for minClusterSize in clusters["minClusterSize"]:
        labels = clusters["minClusterSize"][minClusterSize]["labels"]
        n_cluster = clusters["minClusterSize"][minClusterSize]["n_cluster"]
        clusters["minClusterSize"][minClusterSize]["classification"] = {}


    with open(cluster_path, 'w') as json_file:
        json.dump(clusters, json_file)

    end_time = time.time()
    # Calculate elapsed time
    elapsed_time = end_time - start_time
    # Log elapsed time
    logging.info(f"Clustering {dataset}, Elapsed time: {elapsed_time} seconds")    

    return clusters

# select the point where the 1. derivative is min after the max value.
def find_elbow_point_after_max_change(elbow_y_values):
    max_slope_change = 0
    max_slope_change_index = 0
    for i in range(1, len(elbow_y_values)):
        slope_change = abs(elbow_y_values[i] - elbow_y_values[i-1])
        if slope_change > max_slope_change:
            max_slope_change = slope_change
            max_slope_change_index = i

    min_slope_change = float('inf')
    min_slope_change_index = max_slope_change_index # use max as default if there is no other value
    for i in range(max_slope_change_index + 1, len(elbow_y_values)):
        slope_change = abs(elbow_y_values[i] - elbow_y_values[i-1])
        if slope_change < min_slope_change:
            min_slope_change = slope_change
            min_slope_change_index = i

    return min_slope_change_index

# creates a json for the clustering without the nested structure
def create_cluster_json(linkage_matrix, threshold, criterion):
    clustering = fcluster(linkage_matrix, threshold, criterion=criterion)
    return clustering

# if the linkage data is needed (the split points etc.)
def create_linkage_json(labels, linkage_matrix):
    hierarchy = []
    leaveMax = 0
    for i in range(len(linkage_matrix)):
        cluster1 = int(linkage_matrix[i, 0])
        cluster2 = int(linkage_matrix[i, 1])
        distance = linkage_matrix[i, 2]
        num_observations = int(linkage_matrix[i, 3])

        num_observations1 = 1
        num_observations2 = 1
        for j in range(len(hierarchy)):
            if hierarchy[j]["id"] == cluster1:
                num_observations1 = hierarchy[j]["num_nodes"]

            if hierarchy[j]["id"] == cluster2:
                num_observations2 = hierarchy[j]["num_nodes"]

        # Create a node for cluster1
        # -1 if not a leave, otherwise cluster number
        leave_cluster_1 = -1
        if num_observations1 == 1:
            leave_cluster_1 = int(labels[cluster1])       

        node1 = {'id': cluster1, 'num_nodes': num_observations1, 'leave_cluster': leave_cluster_1}

        leave_cluster_2 = -1
        if num_observations2 == 1:
            leave_cluster_2 = int(labels[cluster2])     
        # Create a node for cluster2
        node2 = {'id': cluster2, 'num_nodes': num_observations2, 'leave_cluster': leave_cluster_2}

        if leave_cluster_1 > leaveMax:
            leaveMax = leave_cluster_1

        if leave_cluster_2 > leaveMax:
            leaveMax = leave_cluster_2
            
        # Create a parent node for cluster1 and cluster2
        parent_node = {'id': i + len(linkage_matrix) + 1, 'distance': distance, 'children': [node1, node2], 'num_nodes': num_observations}

        hierarchy.append(parent_node)

    return hierarchy

# select the most similar tree (centroid) for each cluster based on Banerjee et al.
def selectRepresentativeTree(vectorData, cluster_labels, n_cluster):
    centroid_dict = {}
    # for each cluster find the centroid
    for cluster_id in range(1, n_cluster + 1):
        sum_list = []
        tree_ids = []
        for i in range(len(vectorData)):
            if cluster_labels[i] == cluster_id:
                sum_value = 0
                for j in range(len(vectorData[i])):
                    if i == j:
                        continue
                    sum_value += vectorData[i][j]
                sum_value /= len(vectorData) - 1     
                sum_list.append(sum_value)
                tree_ids.append(i)

        # select tree with lowest value
        index = tree_ids[sum_list.index(min(sum_list))]
        centroid_dict[cluster_id] = index

    return centroid_dict


### Distance Computation ####
def compute_distances_and_extract_rules(distance_path, rule_path, clfs, num_features, class_names, X, tree_classification):
    start_time = time.time()
    vectorData, distances, jsonObj = compute_rule_intervall_sim_and_extract_rules(clfs, num_features, class_names, X, tree_classification, False)
    
    # save distance matrix and json rule obj
    np.save(distance_path, distances, allow_pickle=True)
    
    with open(rule_path , "w") as json_file:
        json.dump(jsonObj, json_file)

    end_time = time.time()
    # Calculate elapsed time
    elapsed_time = end_time - start_time
    # Log elapsed time
    logging.info(f"Distances and Rules {rule_path}, Elapsed time: {elapsed_time} seconds")    
    return vectorData, distances, jsonObj

# A variation of min-max scaling that is less sensitive to outliers is to use the median and 
# the interquartile range (IQR) instead of the mean and range.
def getDataRanges(dataset):
    min_max_values = {}
    counter = 0
    # going through the dataset to see what are the minimum and maximum values for each feature in the dataset. 
    for column in dataset.columns:
        min_value = dataset[column].min()
        max_value = dataset[column].max()
        Q1 = np.percentile(dataset[column], 25)
        Q3 = np.percentile(dataset[column], 75)
        IQR = Q3 - Q1
        # normalized_data = (data - Q1) / IQR
        min_max_values[counter] = {"name": column, "min": min_value, "max": max_value, "IQR": IQR, "Q1": Q1, "Q3": Q3}
        counter += 1

    return min_max_values 

# compute interval width similarity
def interval_width_similarity(interval1, interval2):
    intersection = min(interval1[1], interval2[1]) - max(interval1[0], interval2[0])
    # negative values -> 0
    if intersection <= 0:
        return 0
    length1 = interval1[1] - interval1[0]
    length2 = interval2[1] - interval2[0]
    #based on the intersection of both intervals divided by the max length. 
    return intersection / max(length1, length2)

# compute interval width distance
def interval_width_distance(interval1, interval2):
    intersection = min(interval1[1], interval2[1]) - max(interval1[0], interval2[0])
    # negative values -> 0
    if intersection <= 0:
        return 1  # maximal distance since there's no overlap
    length1 = interval1[1] - interval1[0]
    length2 = interval2[1] - interval2[0]
    similarity = intersection / max(length1, length2)
    # distance is the complement of similarity
    return 1 - similarity

# returns the min interval distance
#saves current min, and break early if it is above
def rule_interval_distance(rule1, rule2, num_features, min_dist):
    # the distance from one set to another set is computed with the interval width 
    total_dist = 0
    for feature_idx in range(num_features):
        interval1 = rule1[feature_idx]
        interval2 = rule2[feature_idx]
        dist = interval_width_distance(interval1, interval2) / num_features  # Divide by num_features here
        total_dist += dist        
        # Optional early exit based on min_dist
        if total_dist > min_dist:
            return min_dist
    return total_dist

def rule_interval_similarity(rule1, rule2, num_features):
    # the distance from one set to another set is computed with the interval width 
    # From the similarity, a distance is computed by subtracting the score from 1 
    total_similarity = 0
    for feature_idx in range(num_features):
        interval1 = rule1[feature_idx]
        interval2 = rule2[feature_idx]
        similarity = interval_width_similarity(interval1, interval2)
        total_similarity += similarity

    return 1 - (total_similarity / num_features)

# extract rules for Max rule visualization
def compute_rule_intervall_sim_and_extract_rules(trees, num_features, class_names, dataset, tree_classification, useRobustMinMaxScaling=False):
    # One rule contains a number of sets of intervals equal to the number of features that are present in the dataset. 
    start_time = time.time()
    num_trees = len(trees)
    min_max_values = getDataRanges(dataset)

    distances = np.zeros((num_trees, num_trees))
    # Create a list to store sets of intervals for each tree
    tree_rules = []
    tree_rules_json = []
    randomForestObj = {}

    """
    The algorithm traverses the decision tree starting from the root in a depth-first search manner. 
    At every step, the sets of intervals are updated. 
    Once the algorithm ends up at a child node of the decision tree, the set of intervals is saved. 
    This means that this set of intervals represents one rule of the decision tree. 
    The algorithm traverses the entire tree until the entire tree is processed, and for each leaf node in the decision tree, a set of intervals is saved
    """
    for i in range(num_trees):
        rule_intervals_json = []
        rule_intervals = {}
        clf = trees[i]
        classification = tree_classification[i]
        feature_intervals = {} 
        # Initialize intervals for each feature
        # prevent missing split features by using the whole range for them
        for f in min_max_values:
            feat = min_max_values[f]
            feature_intervals[f] = (feat["min"], feat["max"])
            if useRobustMinMaxScaling and feat["IQR"] != 0:
                feature_intervals[f] = ((feat["min"] - feat["Q1"]) / feat["IQR"], (feat["max"] - feat["Q1"]) / feat["IQR"])

        def traverse_tree(node_id, feature_intervals):
            feature = clf.tree_.feature[node_id]
            if feature != -2:  # Check if it's not a leaf node
                threshold = clf.tree_.threshold[node_id]
                # Update feature intervals based on the split
                if useRobustMinMaxScaling and min_max_values[feature]["IQR"] != 0:
                    threshold = (threshold - min_max_values[feature]["Q1"]) / min_max_values[feature]["IQR"]
                left_bound = feature_intervals[feature][0]
                right_bound = feature_intervals[feature][1]
                    
                left_interval = (left_bound, threshold)
                right_interval = (threshold, right_bound)

                feature_intervals[feature] = left_interval
                left_child = clf.tree_.children_left[node_id]
                # Recursively traverse left and right children
                traverse_tree(left_child, feature_intervals)

                feature_intervals[feature] = right_interval
                right_child = clf.tree_.children_right[node_id]
                traverse_tree(right_child, feature_intervals)

            else:  # Leaf node, save the intervals as one rule
                value = clf.tree_.value[node_id]
                prediction = class_names[list(value[0]).index(max(value[0]))]
                features = [convert_tuple(feature_intervals[intv]) for intv in feature_intervals]
                
                # get for each leaf node how many test_data samples are correctly and wrongly classified. 
                # for the wrong classification also how often in which class
                correct_predictions = 0
                wrong_predictions = {"count": 0}
                if node_id in classification["correct_predictions_counts"]:
                    correct_predictions = classification["correct_predictions_counts"][node_id]
                
                if node_id in classification["wrong_predictions_counts"]:
                    wrong_predictions = classification["wrong_predictions_counts"][node_id]
                leafNode = {
                    "class": prediction, 
                    "normalizedRangesPerFeature": features, 
                    "correctPredictions": correct_predictions,
                    "wrongPredictions": wrong_predictions
                }
                rule_intervals_json.append(leafNode)
                # the sets of intervals are grouped based on which prediction they belong to. 
                if not prediction in rule_intervals:
                    rule_intervals[prediction] = []
                rule_intervals[prediction].append(features)

        # Start traversal from the root
        root = 0
        traverse_tree(root, feature_intervals)
        tree_rules.append(rule_intervals)
        tree_rules_json.append({"leafNodes": rule_intervals_json, "id": i})


    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Iterate trees, Elapsed time: {elapsed_time} seconds") 
    start_time = time.time()
    # parallel computation of distances
    pool = mp.Pool(mp.cpu_count())
    distances_parallel = {}

    results = pool.starmap(compute_distances_between_trees, [(i, tree_rules, num_trees, num_features) for i in range(num_trees)])
    
    for idx, res in enumerate(results):
        distances_parallel[idx] = res

    pool.close()
    pool.join()


    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"compute distance, Elapsed time: {elapsed_time} seconds") 
    vectorData = []
    # makes symetric matrix and vectorData
    for i in range(num_trees):
        for j in range(i + 1, num_trees):
            distances[i][j] = distances_parallel[i][j]
            distances[j][i] = distances_parallel[i][j]
        tree_rules_json[i]["rule_interval_sim"] = list(distances[i])
        vectorData.append(list(distances[i]))   

    return vectorData, distances, tree_rules_json


def compute_distances_between_trees(tree_index, tree_rules, num_trees, num_features):
    rules = tree_rules[tree_index]
    keys_i = rules.keys()
    distances_for_tree = {}

    for j in range(tree_index + 1, num_trees):
        min_dists = []
        tree_rules_other = tree_rules[j]
        keys_j = tree_rules_other.keys()
        common_keys = keys_i | keys_j

        for classLabel in common_keys:
            # no matching rule for the given class -> edge case (Glass dataset)
            if classLabel not in rules or classLabel not in tree_rules_other:
                min_dists.append(1)
                continue
            rules_1 = rules[classLabel]
            rules_2 = tree_rules_other[classLabel]
            # TODO maybe sort rules in advance? or use nn search
            for r1 in rules_1:
                min_dist = 1
                for r2 in rules_2:
                    # Compute the distance from r1 to r2, using rule_interval_distance
                    min_dist = rule_interval_distance(r1, r2, num_features, min_dist)
                min_dists.append(min_dist)

        avg_dist = np.mean(min_dists)
        distances_for_tree[j] = avg_dist

    return distances_for_tree

def setup_static_folders():
    os.makedirs("static/", exist_ok=True)

    directories = ["clustering", "datasets", "Depth_data", "models", "node_link", "projections", "rules", "trees", "classification"]
    [os.makedirs("static/" + directory, exist_ok=True) for directory in directories]

if __name__ == "__main__":
    setup_static_folders()

    uvicorn.run(app, host="127.0.0.1", port=3030)
