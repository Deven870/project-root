
// Simple state management
let priceUpdate = () => {};
let signalUpdate = () => {};

export const registerPriceCallback = (cb) => {
  priceUpdate = cb;
};

export const registerSignalCallback = (cb) => {
  signalUpdate = cb;
};

export const triggerPriceUpdate = (data) => {
  priceUpdate(data);
};

export const triggerSignalUpdate = (data) => {
  signalUpdate(data);
};
