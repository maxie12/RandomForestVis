import * as d3 from "d3";

// Function to estimate the bandwith
export function silvermansRule(data: any, dimensions: number) {
    const n = data.length;
    const stdDev = d3.deviation(data);
    //@ts-ignore
    return Math.pow((4 * Math.pow(stdDev, 5)) / (3 * n), 1 / 5) * Math.pow((1 / dimensions), 1 / 5);
}

// Function to compute density
export function kernelDensityEstimator(kernel: any, X: any) {
    return function (V: any) {
        return X.map(function (x: number) {
            return [x, d3.mean(V, function (v: number) { return kernel(x - v); })];
        });
    };
}

export function kernelEpanechnikov(k: number) {
    return function (v: number) {
        return Math.abs(v /= k) <= 1 ? 0.75 * (1 - v * v) / k : 0;
    };
}