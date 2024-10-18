class IntermediateOutputsHook:
    def __init__(self):
        self.outputs = []
        self.handles = []
        self.inputs = []
        self.modules = []

    def register(self, model):
        # Register a forward hook for each submodule
        for submodule in model.children():
            if len(list(submodule.children())) == 0:
                handle = submodule.register_forward_hook(self.hook_fn)
                self.handles.append(handle)
            else:
                self.register(submodule)

    def hook_fn(self, module, input, output):
        # Save the intermediate output
        self.outputs.append(output)
        self.inputs.append(input)
        self.modules.append(module)

    def remove_hooks(self):
        # Remove all the registered hooks
        for handle in self.handles:
            handle.remove()