class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        return f'{self.name} says something'
class Dog(Animal):
    def speak(self):
        return f'{self.name} barks'