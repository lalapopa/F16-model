from ambiance import Atmosphere

g = 9.80665

def get_speed_of_sound(height):
    air = Atmosphere(height)
    return air.speed_of_sound

def get_density(height):
    air = Atmosphere(height)
    return air.density
