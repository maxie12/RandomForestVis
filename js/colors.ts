import * as d3 from 'd3'

export const classColorScheme = [
  '#7fcdbb',
  '#41b6c4',
  '#1d91c0',
  '#225ea8',
  '#253494',
  '#081d58',
]

export const classColors = d3.interpolateRgbBasis(classColorScheme);

export const featureColorScheme = [
  '#003f5c',
  '#2f4b7c',
  '#665191',
  '#a05195',
  '#d45087',
  '#f95d6a',
  '#ff7c43',
  '#ffa600',
]


export const featureColors = d3.interpolateRgbBasis(featureColorScheme);
