import xml.etree.ElementTree as xml
import pyproj
import math

class LL2XYProjector:
    def __init__(self, lat_origin, lon_origin):
        self.lat_origin = lat_origin
        self.lon_origin = lon_origin
        self.zone = math.floor((lon_origin + 180.) / 6) + 1
        self.p = pyproj.Proj(proj='utm', ellps='WGS84', zone=self.zone, datum='WGS84')
        [self.x_origin, self.y_origin] = self.p(lon_origin, lat_origin)

    def latlon2xy(self, lat, lon):
        [x, y] = self.p(lon, lat)
        return [x - self.x_origin, y - self.y_origin]

def get_way_ids_by_relation(e, relation_id):
    way_ids = []
    for relation in e.findall('relation'):
        if int(relation.get('id')) == relation_id:
            for member in relation.findall('member'):
                if member.get('type') == 'way':
                    way_ids.append(int(member.get('ref')))
    return way_ids

def get_coordinates_by_way(e, way_id, projector):
    coordinates = []
    way = e.find(f'way[@id="{way_id}"]')
    if way is not None:
        for nd in way.findall('nd'):
            ref_id = int(nd.get('ref'))
            node = e.find(f'node[@id="{ref_id}"]')
            if node is not None:
                lat = float(node.get('lat'))
                lon = float(node.get('lon'))
                x, y = projector.latlon2xy(lat, lon)
                coordinates.append((x, y))
    return coordinates

def find_y_for_x(x_input, e, relation_id, projector):
    way_ids = get_way_ids_by_relation(e, relation_id)
    for way_id in way_ids:
        coordinates = get_coordinates_by_way(e, way_id, projector)
        for i in range(len(coordinates) - 1):
            x1, y1 = coordinates[i]
            x2, y2 = coordinates[i + 1]
            if x1 <= x_input <= x2 or x2 <= x_input <= x1:
                if x1 == x2:
                    return y1  # or y2, both are the same
                # Linear interpolation
                y = y1 + (x_input - x1) * (y2 - y1) / (x2 - x1)
                return y
    return None

def get_y(osm_file, x_input, relation_id, lat_origin=0, lon_origin=0):
    projector = LL2XYProjector(lat_origin, lon_origin)
    e = xml.parse(osm_file).getroot()
    y_output = find_y_for_x(x_input, e, relation_id, projector)
    if y_output is not None:
        return y_output
    else:
        print("No corresponding y coordinate found")



if __name__ == "__main__":
    osm_file = './map.osm' 
    x_input = 1000  
    relation_id = 1000  
    lat_origin = 0 
    lon_origin = 0 

    y_output = get_y(osm_file, x_input, relation_id, lat_origin, lon_origin)

    print(f"y coordinate: {y_output}")
