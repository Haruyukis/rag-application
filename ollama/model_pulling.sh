#!/bin/bash

ollama serve &

sleep 10

ollama pull llama3

ollama run llama3 &