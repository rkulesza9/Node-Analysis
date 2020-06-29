import numpy as np

class Node:
    def __init__(self):
        self.input_nodes = []
        self.output_nodes = []
        
        self.layer = None
        self.layer_in = None
        self.layer_out = None
        
        self.model = None
        self.name = None
        
        self.weight = None
        self.bias = None
        self.activation = None
        self.operation_in = None
        self.operation_out = None
        self.epoch = None
    
    def report(self):
        print(f"NODE [name={self.name} epoch={self.epoch}]")
        print(f"\t model={self.model}")
        print(f"\t layer={self.layer}")
        print()
        print(f"\t layer_in = {self.layer_in}")
        print(f"\t layer_out = {self.layer_out}")
        print()
        print(f"\t weight = {self.weight}")
        print(f"\t bias = {self.bias}")
        print(f"\t activation = {self.activation}")
        print()
        print("IN DEVELOPMENT")
        print(f"\t operation_in = {self.operation_in}")
        print(f"\t operation_out = {self.operation_out}")
        print(f"\t input_nodes = {len(self.input_nodes)}")
        print(f"\t output_nodes = {len(self.output_nodes)}")