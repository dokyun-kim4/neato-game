class person_bbox:
    def __init__(self) -> None:
        self.a = 1
        self.b = 2
        self.c = 4
        self.d = 8
        self.n = 0
    
    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        value = 0
        stop = False
        match self.n:
            case 0:
                value = self.a
            case 1:
                value = self.b
            case 2:
                value = self.c
            case 3:
                value = self.d
            case _:
                stop = True

        if stop:
            raise StopIteration

        self.n += 1
        return value
    
bbox = person_bbox()

for value in bbox:
    print(value)