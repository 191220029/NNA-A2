pub mod linear;
mod initialize;

trait Module {
    fn init(&mut self):
        self.training = True

    def parameters(self) -> List["Tensor"]:
        return Parameter._unpack_params(self.__dict__)
    
    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    @abstractmethod
    def forward(self):
        pass

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True
}