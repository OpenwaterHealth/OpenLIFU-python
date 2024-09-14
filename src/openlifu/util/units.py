def getunittype(unit):
    unit = unit.lower()
    if unit in ['micron', 'microns']:
        return 'distance'
    elif unit in ['minute', 'minutes', 'min', 'mins', 'hour', 'hours', 'hr', 'hrs', 'day', 'days', 'd']:
        return 'time'
    elif unit in ['rad', 'deg', 'radian', 'radians', 'degree', 'degrees', '°']:
        return 'angle'
    elif 'sec' in unit:
        return 'time'
    elif 'meter' in unit:
        return 'distance'
    elif 'micron' in unit:
        return 'distance'
    elif unit.endswith('s'):
        return 'time'
    elif unit.endswith('m'):
        return 'distance'
    elif unit.endswith('m2') or unit.endswith('m^2'):
        return 'area'
    elif unit.endswith('m3') or unit.endswith('m^3'):
        return 'volume'
    elif unit.endswith('hz'):
        return 'frequency'
    else:
        return 'other'

def getunitconversion(from_unit, to_unit, unitratio=None, constant=None):
    if not from_unit:
        return 1.0

    if unitratio is not None and constant is not None:
        if '/' not in unitratio:
            raise ValueError('Conversion unit ratio must have a \'/\' symbol')

        unitn, unitd = unitratio.split('/')
        type0 = getunittype(from_unit)
        type1 = getunittype(to_unit)
        typen = getunittype(unitn)
        typed = getunittype(unitd)

        if type0 == typed and type1 == typen:
            scl = getunitconversion(from_unit, unitd) * constant * getunitconversion(unitn, to_unit)
        elif type0 == typen and type1 == typed:
            scl = getunitconversion(from_unit, unitn) * 1 / constant * getunitconversion(unitd, to_unit)
        elif type0 == type1:
            scl = getunitconversion(from_unit, to_unit)
        else:
            raise ValueError('Unit type mismatch {} -> ({}/{}) -> {}'.format(type0, typen, typed, type1))
    else:
        slash0 = from_unit.find('/')
        slash1 = to_unit.find('/')

        if slash0 != -1 and slash1 != -1:
            num0 = from_unit[:slash0]
            denom0 = from_unit[slash0+1:]
            num1 = to_unit[:slash1]
            denom1 = to_unit[slash1+1:]
            scl = getunitconversion(num0, num1) / getunitconversion(denom0, denom1)
        elif slash0 == -1 and slash1 == -1:
            type0 = getunittype(from_unit)
            type1 = getunittype(to_unit)

            if type0 != type1:
                raise ValueError('Unit type mismatch ({}) vs ({})'.format(type0, type1))

            if type0 == 'other':
                if from_unit[-1] != to_unit[-1]:
                    raise ValueError('Cannot convert {} to {}'.format(from_unit, to_unit))

                i = 0
                while i < min(len(from_unit), len(to_unit)) and from_unit[-i:] == to_unit[-i:]:
                    type = from_unit[-i:]
                    i += 1

                scl0 = getsiscale(from_unit, type)
                scl1 = getsiscale(to_unit, type)
                scl = scl0 / scl1
            else:
                scl0 = getsiscale(from_unit, type0)
                scl1 = getsiscale(to_unit, type0)
                scl = scl0 / scl1
        else:
            raise ValueError('Unit ratio mismatch ({} vs {})'.format(from_unit, to_unit))

    return scl

def getsiscale(unit, type):
    type = type.lower()

    if type in ['distance', 'area', 'volume']:
        idx = unit.find('meters')
        if idx == -1:
            idx = unit.find('meter')
            if idx == -1:
                if unit.lower() == 'micron':
                    idx = 6
                else:
                    idx = unit.rfind('m')
                    if idx == -1:
                        idx = len(unit)

    elif type == 'time':
        idx = unit.find('seconds')
        if idx == -1:
            idx = unit.find('second')
            if idx == -1:
                idx = unit.find('sec')
                if idx == -1:
                    idx = unit.rfind('s')
                    if idx == -1:
                        idx = len(unit)

    elif type == 'angle':
        idx = len(unit)

    elif type == 'frequency':
        idx = len(unit) - 2

    else:
        idx = len(unit) - len(type) + 1

    idx = idx
    prefix = unit[:idx]

    if not prefix:
        scl = 1.0
    else:
        scl = 1.0

        if prefix == 'pico' or prefix == 'p':
            scl = 1.0e-12
        elif prefix == 'nano' or prefix == 'n':
            scl = 1.0e-9
        elif prefix == 'micro' or prefix == 'u' or prefix == '\u00b5' or prefix == '\u03bc':
            scl = 1.0e-6
        elif prefix == 'milli' or prefix == 'm':
            scl = 1.0e-3
        elif prefix == 'centi' or prefix == 'c':
            scl = 1.0e-2
        elif prefix == '':
            scl = 1.0
        elif prefix == 'kilo' or prefix == 'k':
            scl = 1.0e3
        elif prefix == 'mega' or prefix == 'M':
            scl = 1.0e6
        elif prefix == 'giga' or prefix == 'G':
            scl = 1.0e9
        elif prefix == 'tera' or prefix == 'T':
            scl = 1.0e12
        elif prefix == 'min' or prefix == 'minute':
            scl = 60.0
        elif prefix == 'hour' or prefix == 'hr':
            scl = 60.0 * 60.0
        elif prefix == 'day' or prefix == 'd':
            scl = 60.0 * 60.0 * 24.0
        elif prefix == 'rad' or prefix == 'radian' or prefix == 'radians':
            scl = 1.0
        elif prefix == 'deg' or prefix == 'degree' or prefix == 'degrees' or prefix == '\u00b0':
            scl = 2 * 3.14159265358979323846 / 360
        else:
            if prefix:
                raise ValueError('Unknown prefix {}'.format(prefix))

    if type == 'area':
        scl = scl ** 2.0
    elif type == 'volume':
        scl = scl ** 3.0

    return scl
