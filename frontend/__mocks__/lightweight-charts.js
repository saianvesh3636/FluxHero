/**
 * Mock for lightweight-charts library
 * Used in Jest tests to avoid issues with canvas rendering
 */

const createChart = jest.fn(() => ({
  addSeries: jest.fn(() => ({
    setData: jest.fn(),
  })),
  timeScale: jest.fn(() => ({
    fitContent: jest.fn(),
  })),
  applyOptions: jest.fn(),
  remove: jest.fn(),
}));

const LineSeries = {};

module.exports = {
  createChart,
  LineSeries,
};
