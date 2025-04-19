import xml.etree.ElementTree as ET
from pathlib import Path


def load_osm_file(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    return root, tree

def save_osm_file(root, tree, filename):
    tree.write(filename)

def delete_nodes_and_related_ways(root, node_ids):
    # Collect ways to delete
    ways_to_delete = set()
    
    # Find all ways that reference the nodes to be deleted
    for way in root.findall('way'):
        for nd in way.findall('nd'):
            if int(nd.get('ref')) in node_ids:
                ways_to_delete.add(way.attrib['id'])
                break
    
    # Delete the ways
    for way_id in ways_to_delete:
        way = root.find(f".//way[@id='{way_id}']")
        if way is not None:
            root.remove(way)
            print(f"Deleted way {way_id}")
    
    # Delete the nodes
    for node_id in node_ids:
        node = root.find(f".//node[@id='{node_id}']")
        if node is not None:
            root.remove(node)
            print(f"Deleted node {node_id}")

def delete_relations_related_to_nodes_or_ways(root, node_ids, way_ids):
    relations_to_delete = set()
    
    # Find all relations that reference the nodes or ways to be deleted
    for relation in root.findall('relation'):
        for member in relation.findall('member'):
            ref_id = int(member.get('ref'))
            if ref_id in node_ids or ref_id in way_ids:
                relations_to_delete.add(relation.attrib['id'])
                break
    
    # Delete the relations
    for relation_id in relations_to_delete:
        relation = root.find(f".//relation[@id='{relation_id}']")
        if relation is not None:
            root.remove(relation)
            print(f"Deleted relation {relation_id}")

def add_custom_relation(root, relation_id, relation_type, way_members, version=1):
    # Create a new relation element with version attribute
    relation = ET.Element('relation', id=str(relation_id), visible='true', version=str(version))
    type_tag = ET.SubElement(relation, 'tag', k='type', v=relation_type)
    
    # Add ways to the relation
    for way in way_members:
        member_element = ET.SubElement(relation, 'member', 
                                       type='way', 
                                       ref=str(way['ref']), 
                                       role=way.get('role', ''))
    
    # Add the relation to the root element
    root.append(relation)
    print(f"Added relation {relation_id} with type '{relation_type}' and version {version}")



def delete_all_relations(root):
    for relation in root.findall('relation'):
        root.remove(relation)
        print(f"Deleted relation {relation.attrib['id']}")

def delete_relation_by_id(root, relation_id):
    """
    删除指定 relation id 的关系
    """
    relation = root.find(f".//relation[@id='{relation_id}']")
    if relation is not None:
        root.remove(relation)
        print(f"Deleted relation {relation_id}")
    else:
        print(f"Relation {relation_id} not found.")

def delete_unrelated_nodes_and_ways(root, relation_ids):
    # 找到所有与指定关系 ID 相关的节点和道路
    related_nodes = set()
    related_ways = set()

    # 查找与指定关系 ID 相关的节点和道路
    for relation in root.findall('relation'):
        if int(relation.attrib['id']) in relation_ids:
            for member in relation.findall('member'):
                ref_id = int(member.get('ref'))
                if member.get('type') == 'node':
                    related_nodes.add(ref_id)
                elif member.get('type') == 'way':
                    related_ways.add(ref_id)

    # 查找所有与指定的 way 相关的节点
    for way in root.findall('way'):
        if int(way.attrib['id']) in related_ways:
            for nd in way.findall('nd'):
                related_nodes.add(int(nd.attrib['ref']))

    # 删除所有不相关的节点
    for node in root.findall('node'):
        if int(node.attrib['id']) not in related_nodes:
            root.remove(node)
            print(f"Deleted node {node.attrib['id']}")

    # 删除所有不相关的道路
    for way in root.findall('way'):
        if int(way.attrib['id']) not in related_ways:
            root.remove(way)
            print(f"Deleted way {way.attrib['id']}")



def extract_elements_as_set(root):
    return set(ET.tostring(child) for child in root)

def is_root_equal_by_elements(file1, file2):
    root1 = ET.parse(file1).getroot()
    root2 = ET.parse(file2).getroot()
    return extract_elements_as_set(root1) == extract_elements_as_set(root2)

if __name__ == "__main__":

    # local file path
    base_dir = Path(__file__).parent.resolve()
    filepath = (base_dir / "../../maps/DR_CHN_Merging_ZS.osm").resolve()
    saved_filepath = (base_dir / "../../maps/merge.osm").resolve()

    ids, types, members = [], [], []
    ######################### customise relations for the map you want to research ############################
    custom_relation_id = 1000
    custom_relation_type = "line1"
    custom_way_members = [
        {'ref': 10062, 'role': 'outer'},
        {'ref': 10046, 'role': 'outer'},
        {'ref': 10047, 'role': 'outer'},
        {'ref': 10018, 'role': 'outer'},
        {'ref': 10048, 'role': 'outer'},
        {'ref': 10042, 'role': 'outer'},
        {'ref': 10008, 'role': 'outer'},
        {'ref': 10012, 'role': 'outer'},
    ]

    ids.append(custom_relation_id)
    types.append(custom_relation_type)
    members.append(custom_way_members)


    custom_relation_id = 2000
    custom_relation_type = "line2"
    custom_way_members = [
        {'ref': 10045, 'role': 'outer'},
        {'ref': 10027, 'role': 'outer'},
        {'ref': 10024, 'role': 'outer'},
        {'ref': 10030, 'role': 'outer'},
        {'ref': 10035, 'role': 'outer'},
        {'ref': 10017, 'role': 'outer'},
        {'ref': 10033, 'role': 'outer'},
        {'ref': 10013, 'role': 'outer'},
    ]
    ids.append(custom_relation_id)
    types.append(custom_relation_type)
    members.append(custom_way_members)


    custom_relation_id = 3000
    custom_relation_type = "line3"
    custom_way_members = [
        {'ref': 10044, 'role': 'outer'},
        {'ref': 10028, 'role': 'outer'},
        {'ref': 10022, 'role': 'outer'},
        {'ref': 10031, 'role': 'outer'},
        {'ref': 10034, 'role': 'outer'},
        {'ref': 10016, 'role': 'outer'},
        {'ref': 10023, 'role': 'outer'},
        {'ref': 10014, 'role': 'outer'},
    ]
    ids.append(custom_relation_id)
    types.append(custom_relation_type)
    members.append(custom_way_members)


    custom_relation_id = 4000
    custom_relation_type = "line4"
    custom_way_members = [
        {'ref': 10043, 'role': 'outer'},
        {'ref': 10029, 'role': 'outer'},
        {'ref': 10015, 'role': 'outer'},
        {'ref': 10032, 'role': 'outer'},
        {'ref': 10036, 'role': 'outer'},
        {'ref': 10059, 'role': 'outer'},
        {'ref': 10023, 'role': 'outer'},
        {'ref': 10014, 'role': 'outer'},
    ]
    ids.append(custom_relation_id)
    types.append(custom_relation_type)
    members.append(custom_way_members)

    relation_ids_to_keep = ids

    ############################################## load OSM file ###########################################
    root, tree = load_osm_file(filepath)

    delete_all_relations(root)
    
    # #添加自定义关系
    for custom_relation_id, custom_relation_type, custom_way_members in zip(ids, types, members):
        add_custom_relation(root, custom_relation_id, custom_relation_type, custom_way_members, version=1)


    delete_unrelated_nodes_and_ways(root, relation_ids_to_keep)

    # save file to target path
    save_osm_file(root, tree, saved_filepath)


