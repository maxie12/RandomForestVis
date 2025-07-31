import { readData, readDtreeCSVData, readCluster } from "./DataLoader";
import * as PIXI from 'pixi.js';
import { Viewport } from 'pixi-viewport';
import Split from 'split.js'
import { Cluster, FeatureMetaData, LinkageSplitPoint } from "./types";
import { generateBarcodesForTrees } from "./Barcode/Barcode";
import { generateFeaturePlots } from "./Featureplot/Featureplot";
import axios from 'axios'

import { createProjection, selectClusterProjection } from "./Projection";
import { createInputSliders } from "./InputSlider";
import { createCharts } from "./Sidebar/Sidebar";
import { setContainerPositionsAndScale } from "./LinearLayout";
import { initFeatureDistribution, selectCluster, updateAll } from "./Interactions";
import { initializeTrees, positionDecisionTrees, showDecisionTreesForCluster } from "./NodeLink";
import { printDuration, printStartTime } from "./Timing";
import { initializeBarCodeToolTip } from "./Barcode/Tooltip";

// Initialize split panel
Split(['#feature-plot', '#detail-view'], { onDrag: updateVisPositions, sizes: [61.8, 38.2] })//TODO: Set scrollbar to be independent per view.
// pixi.js drawing

//////////////////////////
// main plot
///////////////////////////
const mainPlotWidth = 1200;


let mainApp: PIXI.Application;
export let viewPort: Viewport;


// only for first load
const urls = [
  'http://localhost:3030/get_data'
];

// Array to store the promises for each request
const requests: Promise<any>[] = [];
let dataset: String = "penguins_max_depth4"; //Initial dataset for the evaluation
// let dataset: String = "Glass-clean";

// Create a promise for each request
urls.forEach(url => {
  /*if (url === 'http://localhost:3030/get_projection') {
    requests.push(axios.post(url, { projection, dataset }));
  } else {
    requests.push(axios.post(url, dataset));
  }*/
  requests.push(axios.post(url, dataset));
});

// Concurrent requests using axios.all
axios.all(requests)
  .then(axios.spread((...responses) => {
    responses.forEach((response, index) => {
      if (index == 0) {
        init(response.data)
      }
    });
  }))
  .catch(error => {
    console.error('Error in one or more requests:', error);
  });


//data upload
document.addEventListener("DOMContentLoaded", () => {
  const fileInput = document.getElementById('fileInput');
  // optional chaining in case fileInput, event, target or files is undefined
  fileInput?.addEventListener('change', async (event) => {
    const file = (event?.target as HTMLInputElement)?.files?.[0];
    const formData = new FormData();
    formData.append('file', file ?? ''); // Providing an empty string as default value if file is null or undefined
    showLoading();
    try {
      const response = await axios.post('http://localhost:3030/upload_data', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      init(response.data)
      hideLoading();
    } catch (error) {
      hideLoading();
      console.error(error);
    }
  });
});


function init(data: any) {
  printStartTime("Full process")
  asyncInit(data);
}

/**
 * 
 * @param data 
 * @param clusterValue if not undefined, use the clustervalue as the data point and don't rerender the sidebar.
 */
export async function asyncInit(data: any, clusterValue: number | undefined = undefined) {
  if (mainApp) {
    //clean up all previous containers. Keep the stage and mainapp intact to prevent webgl errors
    for (let container of mainApp.stage.children) {
      container.destroy({ children: true, texture: true });
    }
  }
  if (!mainApp) {
    mainApp = new PIXI.Application();
    await mainApp.init({
      width: mainPlotWidth,
      height: 1080,
      autoDensity: true,
      backgroundAlpha: 0,
      antialias: true,
      resolution: 2,
      canvas: document.querySelector('#feature-plot canvas') as HTMLCanvasElement
    });
  }

  viewPort = new Viewport({
    screenWidth: mainPlotWidth,
    screenHeight: 1080,
    worldWidth: 1000,
    worldHeight: 1000,
    events: mainApp.renderer.events,
  })
  mainApp.stage.addChild(viewPort);

  viewPort
    .drag()
    .wheel()
    .decelerate()


  let cluster = readCluster(data.cluster.minClusterSize[data.cluster["elbow"]]);
  if (clusterValue != undefined) { //still need to initialize them
    cluster = readCluster(data.cluster.minClusterSize[clusterValue]);
  }


  prepareDepthAndBarPlot(data.rules, data.depth_data, cluster, data["features"], data.classes);

  if (clusterValue == undefined) { //still need to initialize them
    createInputSliders(data);
    createCharts(data.features, data.classes, data.classification);
    initFeatureDistribution(Object.entries(data["features"]).length);
  }

  createProjection(data.projection, cluster);
  initializeTrees(data.trees, cluster, data.features, data.classes);

  //set position of the main view
  let width = document.getElementById("feature-plot")!.offsetWidth! - 30; //-30 to account for scrollbars
  let height = document.getElementById("sidebar")!.offsetHeight! - 100; //TODO: Get the proper available height without scroll
  setContainerPositionsAndScale(Array.from(representativeContainers.values()), width, height);


  updateVisPositions()

  if (clusterValue != undefined) {
    let clusterId = 1;
    let centroidId = cluster.centroids.get(clusterId)!;
    showDecisionTreesForCluster(centroidId, clusterId, cluster);

    //Update all visualizations such that they show highlighting in case it was selected
    updateAll();
  }

  printDuration("Full process")
}


//Functions for plot initialization

let mainPlotStage: PIXI.Container;
let featurePlotContainers: Map<number, PIXI.Container> //RepresentativeTreeId,container
let barcodeContainers: Map<number, PIXI.Container> //RepresentativeTreeId,container
let representativeContainers: Map<number, PIXI.Container> //RepresentativeTreeId,container that has bot a featurePlot and a barcode.
let barCodeToolTipContainer: PIXI.Container;

export function prepareDepthAndBarPlot(rules: object, depth_data: any, cluster: Cluster, features: FeatureMetaData, classes: any) {
  printStartTime("read DTreeData")
  let dTreeData = readDtreeCSVData(depth_data, Object.keys(classes).length);
  printDuration("read DTreeData")
  featurePlotContainers = generateFeaturePlots(dTreeData, cluster, 450, features);

  printStartTime("read rulesData")
  let trees = readData(rules);
  printDuration("read rulesData")

  barcodeContainers = generateBarcodesForTrees(trees, cluster, classes);

  representativeContainers = new Map();
  for (let [repId, featureContainer] of featurePlotContainers) {
    let barcodeContainer = barcodeContainers.get(repId)!;
    let representativeContainer = new PIXI.Container();
    representativeContainer.addChild(barcodeContainer);
    representativeContainer.addChild(featureContainer);
    representativeContainers.set(repId, representativeContainer);

    representativeContainer.interactive=true;
    representativeContainer.onclick = function () {
      // select cluster for node link
      selectCluster(repId);
    }



    let fontSize = 15;

    let outlineOffSet = 5;
    let topOffset = fontSize + 2 + outlineOffSet / 2;
    let barcodeYOffset = 5;

    let targetHeight = featureContainer.getBounds().height - topOffset;
    let targetWidth = featureContainer.getBounds().width - outlineOffSet;

    let barCodeXScale = targetWidth / barcodeContainer.getBounds().width;
    let barCodeYScale = targetHeight / barcodeContainer.getBounds().height;

    let featureXScale = targetWidth / featureContainer.getBounds().width;
    let featureYScale = targetHeight / featureContainer.getBounds().height;

    //Scale both to allow for the offset for the selection rectangle.
    //start the barcode a bit below the feature, and scale to same width and height
    featureContainer.x = outlineOffSet / 2
    featureContainer.y = topOffset;
    featureContainer.scale.set(featureXScale, featureYScale);

    barcodeContainer.x = outlineOffSet / 2
    barcodeContainer.y = topOffset + targetHeight + barcodeYOffset;
    barcodeContainer.scale.set(barCodeXScale, barCodeYScale);
    // featureContainer.setTransform(outlineOffSet / 2, topOffset, featureXScale, featureYScale);
    // barcodeContainer.setTransform(outlineOffSet / 2, topOffset + targetHeight + barcodeYOffset, barCodeXScale, barCodeYScale);
    representativeContainer.addChild(addOutlineObject(repId, targetWidth, targetHeight, topOffset, outlineOffSet, barcodeYOffset));
    let textString = "Cluster " + repId;
    let text = new PIXI.Text({ text: textString, style: { fontSize: fontSize } });
    text.anchor.set(0.5, 0)
    text.position.set((targetWidth + outlineOffSet) / 2, 0)
    representativeContainer.addChild(text);

    if (representativeContainers.size === 1) {
      selectClusterMainView(repId);
      selectClusterProjection(repId);
    }
  }


  mainPlotStage = new PIXI.Container();
  representativeContainers.forEach((repContainer, treeId, map) => {
    mainPlotStage.addChild(repContainer);
  });
  //position the clusters
  updateVisPositions();

  //initialize barcode tooltip
  barCodeToolTipContainer = initializeBarCodeToolTip(mainPlotStage);

  viewPort.addChild(mainPlotStage);
}


let outlines = new Map<number, PIXI.Container>()
let currentSelection: number | undefined = undefined;
function addOutlineObject(id: number, targetWidth: number, targetHeight: number, topOffset: number, offset: number, barcodeYOffset: number): PIXI.Graphics {
  let width = targetWidth + offset;
  let height = targetHeight * 2 + barcodeYOffset + offset;
  let lineWidth = offset / 2 / 2; // Division by 2 as there is space on both sides. Another division by 2 to leave some space
  let halfLineWidth = lineWidth / 2;
  let outlineObject = new PIXI.Graphics();
  // outlineObject.lineStyle(lineWidth, "#000000", 1)

  outlineObject
    .moveTo(halfLineWidth, halfLineWidth)
    .lineTo(halfLineWidth, height - halfLineWidth + topOffset)
    .lineTo(width - halfLineWidth, height - halfLineWidth + topOffset)
    .lineTo(width - halfLineWidth, halfLineWidth)
    .lineTo(halfLineWidth, halfLineWidth)
  outlineObject.stroke({ width: lineWidth, color: "#000000" })

  outlineObject.alpha = 0;
  outlines.set(id, outlineObject);
  return outlineObject;
}

export function selectClusterMainView(treeId: number) {
  if (currentSelection !== undefined) {
    outlines.get(currentSelection)!.alpha = 0;
  }
  outlines.get(treeId)!.alpha = 1;
  currentSelection = treeId;
}

let currentHoverSelection: number | undefined = undefined
export function hoverClusterMainView(treeId: number) {
  if (treeId == currentSelection) {//don't need to do anything if it is the currently selected tree
    return;
  }
  //deselect the old one
  if (currentHoverSelection !== undefined) {
    outlines.get(currentHoverSelection)!.alpha = 0;
  }
  outlines.get(treeId)!.alpha = 0.8;
  currentHoverSelection = treeId;
}

export function dehoverClusterMainView(treeId: number) {
  if (treeId == currentSelection) {//don't need to do anything if it is the currently selected tree
    return;
  }
  //if we already changed the selection, no need to do anything
  if (treeId !== currentHoverSelection) {
    return;
  }
  if (currentHoverSelection !== undefined) {
    outlines.get(currentHoverSelection)!.alpha = 0;
  }
}

let currentMainPlotWidth: number;
export function getCurrentMainPlotWidth() { return currentMainPlotWidth };

function updateVisPositions() {
  //set the decision tree cluster view
  let width = document.getElementById("feature-plot")!.offsetWidth! - 30; //-30 to account for scrollbars
  let height = document.getElementById("sidebar")!.offsetHeight! - 100; //TODO: Get the proper available height without scroll
  mainApp.renderer.resize(width, height);
  viewPort.resize(width, height);

  currentMainPlotWidth = width; //update the width so we can use it in tooltips
}

// Function to show loading animation
export function showLoading() {
  const loading = document.getElementById('loading-overlay');
  if (loading) {
    loading.style.display = 'block';
  }
}

// Function to hide loading animation
export function hideLoading() {
  const loading = document.getElementById('loading-overlay');
  if (loading) {
    loading.style.display = '';
  }
}



// --- Only for evaluation ---
enum DATASET { PENGUIN = "Penguin", GLASS = "Glass" }

let currentDataset = DATASET.PENGUIN;
//click on the entire label
document.getElementById('penguinLabel')!.addEventListener('click', function (event) {
  clickDataset(event, DATASET.PENGUIN)
});

//click on the entire label
document.getElementById('glassLabel')!.addEventListener('click', function (event) {
  clickDataset(event, DATASET.GLASS);
});

function clickDataset(event: any, dataset: DATASET) {
  if (dataset == currentDataset) {
    return;
  }
  currentDataset = dataset;

  //uncheck all
  (document.getElementById('penguin') as HTMLInputElement).checked = false;
  (document.getElementById('glass') as HTMLInputElement).checked = false;

  //check the one we clicked on and update the visualization
  if (dataset == DATASET.PENGUIN) {
    (document.getElementById('penguin') as HTMLInputElement).checked = true;
    updateDataset('penguins_max_depth4');

  }
  if (dataset == DATASET.GLASS) {
    (document.getElementById('glass') as HTMLInputElement).checked = true;
    updateDataset('Glass-clean');
  }
}

function enableAllInteractions(container: any) {
  container.children.forEach((child: any) => {
    child.interactive = true; // Disable interaction for this child
    if (child.children.length > 0) {
      enableAllInteractions(child); // Recursively disable interactions for child containers
    }
  });
}

function disableAllInteractions(container: any) {
  container.children.forEach((child: any) => {
    child.interactive = false; // Disable interaction for this child
    if (child.children.length > 0) {
      disableAllInteractions(child); // Recursively disable interactions for child containers
    }
  });
}

function updateDataset(selectedDataset: string) {
  showLoading();
  axios.post('http://localhost:3030/get_data', selectedDataset)
    .then(response => {

      init(response.data);
      // /*
      // Disable interactions and enable them after timeout because of eventhandler pixi js error 
      // Cannot find propagation path to disconnected target
      // */
      // mainPlotStage.interactive = false;
      // disableAllInteractions(mainPlotStage)
      // // Add a timeout
      // setTimeout(() => {
      //   hideLoading();

      //   mainPlotStage.interactive = true;
      //   enableAllInteractions(mainPlotStage)
      // }, 1000);
      hideLoading();
    })
    .catch(error => {
      hideLoading();
      console.error("Error loading dataset:", error);
    });

}