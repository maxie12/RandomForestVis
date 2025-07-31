import * as PIXI from 'pixi.js';

export interface LeafNode { //Represents a path down the decision trees
    normalizedRangesPerFeature: [number, number][]; //[min,max] interval per feature
    class: string; //Which class data that goes down in this leaf is encoded to.
    representedLeafNodes: LeafNode[] //which other leafnodes this leafnode is representing
    correctPredictions: number; //Number of test instances where this leaf correctly predicts the value
    wrongPredictions: wrongPredictions; // holds which predictions were wrongly made
    allPredictions?: Predictions; //Set after merging. Holds all predictions in the complete matrix
    allRepresentedPredictions?: Predictions; //Set after merging. Holds all predictions including those from it's representatives if any.
}

export interface wrongPredictions { //represents the (wrong) predictions
    count: number //number of total wrong predictions
    [originalClassId: number]: { //original class of test case
        [classifiedAsId: number]: number, //misclassified as this case
        count: number // amount of times this got misclassified
    }

}

export interface Predictions { //represents the (wrong) predictions
    count: number //number of total wrong predictions
    [originalClassId: number]: { //original class of test case
        [classifiedAsId: number]: number, //misclassified as this case
        count: number // amount of times this got misclassified
    }

}

export interface leafPredictions { //represents the (wrong) predictions or a leafnode
    count: number //number of total wrong predictions
    [originalClassId: number]: { //original class of test case
        [classifiedAsId: number]: number, //misclassified as this case
        count: number // amount of times this got misclassified
    }

}

export interface Classifications {
    accuracy: number,
    agreement: number[][],
    correctPredictions: {
        count: number,
        [classId: number]: number
    }
    wrongPredictions: Predictions
}

export interface Classes {
    [className: string]: {
        count: number, //how often this class is present in the (input?) data
        index: number //Which index this class has
    }
}

export interface Tree {
    id: number; //id of the decision tree
    leafNodes: LeafNode[]; //all the leafnodes of the decision trees. 
    clusterDistances: number[];
}

export interface DecisionTree {
    id: number; //id of the decision tree
    leafNodes: LeafNode[]; //all the leafnodes of the decision trees. 
    clusterDistances: number[];
}

export interface DecisionTreeNode {
    accuracy: number;
    ancestorFeatures: any;
    children: DecisionTreeNode[];
    correctPredictions: number
    correctPredictionsClassesCounts: {
        [featureName: string]: number
    }
    feature: string;
    node_id: number
    orientation: string
    threshold: number | string;
    values: number[];
    wrongPredictions: Predictions
    allPredictions?: Predictions //Calculated in front end for now, could be calculated on backend for speed as well..
    isGrey: boolean; //used for efficiency of dimming
}

export interface DecisionTreeLeaf extends DecisionTreeNode {
    correctPredictions: number;
    wrongPredictions: Predictions
}

export interface Cluster {
    centroids: Map<number, number>, //Key holds the clusterId, number holds the id of the tree which is the centroid.
    labels: number[], //For each treeid, holds which clusterId representedi
    //linkage_Data: LinkageSplitPoint[],
    n_cluster: number
}

export interface LinkageSplitPoint {
    id: number //identifier of the split point
    distance: number //Distance to the leaf node
    num_nodes: number
    children: [{ //always 2 children, can be empty.
        id: number //identifier of a split point
        num_nodes: number
        leave_cluster: number
    },
        {
            id: number //identifier of a split point
            num_nodes: number
            leave_cluster: number
        }]
}

export interface LinkageTree {
    id: number
    distance: number
    child1: LinkageTree | number
    child2: LinkageTree | number
}


export interface DTree {
    id: number; //id of the tree
    rootNode?: DTreeNode;
    nodeMap: Map<number, DTreeNode>; // nodeID -> DTreeNode
}

export interface DTreeNode {
    id: number; //id of the node
    isLeaf: boolean;
    //nrSamples?: number; //Percentage of paths that follow this branch?
    //impurity?: number; //?
    splitFeature?: string; //Split on feature with name {string}
    threshold?: number; //unnormalized threshold value
    children: number[]; //left and right child id. Id's are unique per tree, but not for all trees.
    depth: number;
    correctPredictions?: number;
    wrongPredictions?: Predictions;
    allPredictions: Predictions
}

export interface FeaturePlotData {
    /**TreeID */
    id: number;
    /** level -> (featureName, count) tuples for each feature that is split on */
    levelFeatures: Map<number, [string, number][]>;
    /**level -> number of features that are split on in this level*/
    levelTotalFeatures: Map<number, number>;
    /**level -> feature being split on -> rules. Holds for each level and feature, which rules these features are involved in. */
    featureLevelRules: Map<number, Map<string, Rule[]>>;
}

export interface FeatureMetaData {

    [featureName: string]:
    {
        index: number
        is_categorical: boolean;
        numerical_categories: number[]
        categories_norminal: number[]
        range: [number, number] /**First value has the minimum, second has the maximum */
        scaled_values: [number, number]
    }
}


export interface Rule {
    ranges: [number, number][]
    predictions: Predictions
}

export interface DimRectangle{
    normalizedRanges: [number, number][],
    predictions: Predictions,
    rectangles: PIXI.Graphics[]
}