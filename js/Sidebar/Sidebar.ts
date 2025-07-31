import { classColors, featureColors } from "../colors";
import { Classifications, FeatureMetaData } from "../types";
import { createClassChartMatrix } from "./ClassChart";
import { createFeatureChart } from "./FeatureChart";

export function createCharts(features: FeatureMetaData, classes: any, classification: Classifications) {
    createClassChartMatrix("#classes-collapse", classes, classColors, classification);
    createFeatureChart("#features-collapse", features, featureColors);
}
