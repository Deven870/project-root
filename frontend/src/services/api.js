import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
});

export const fetchPrice = async (symbol) => {
  const response = await api.get(`/api/prices/${symbol}`);
  return response.data;
};

export const fetchBatchPrices = async (symbols) => {
  const response = await api.get(`/api/prices/batch?symbols=${symbols.join(',')}`);
  return response.data;
};

export const fetchSignal = async (symbol) => {
  const response = await api.get(`/api/signals/${symbol}`);
  return response.data;
};

export const fetchPortfolio = async () => {
  const response = await api.get('/api/portfolio');
  return response.data;
};

export const fetchStocks = async () => {
  const response = await api.get('/api/stocks');
  return response.data;
};

export const fetchHealth = async () => {
  const response = await api.get('/health');
  return response.data;
};

export default api;
