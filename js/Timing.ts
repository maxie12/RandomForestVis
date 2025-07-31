
let timers: Map<string, number> = new Map();

let debugTimer = false;

export function printStartTime(timerName: string) {
    let startTime = (new Date()).getTime()
    timers.set(timerName, startTime);
}

export function printDuration(timerName: string) {
    let endTime = (new Date()).getTime()
    let startTime = timers.get(timerName);
    if (startTime == undefined) {
        throw new Error("No timer with name " + timerName + "defined.")
    }
    let duration = endTime - startTime;
    if (debugTimer) {
        console.log("Timer " + timerName + " has a duration of " + ((duration) / 1000) + " seconds");
    }
}