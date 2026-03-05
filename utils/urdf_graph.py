import xml.etree.ElementTree as ET
from collections import deque, defaultdict


class UrdfGraph:
    def __init__(self, urdf_path):
        self.urdf_path = urdf_path
        self.tree = ET.parse(urdf_path)
        self.root = self.tree.getroot()

        self._parent_to_children_joints = defaultdict(list)
        self._parent_to_children_links = defaultdict(list)
        self._child_links = set()
        self._parent_links = set()
        self._joints = []

        self._build_graph()

    def _build_graph(self):
        for j in self.root.findall("joint"):
            name = j.get("name")
            jtype = j.get("type")
            if jtype == "fixed":
                continue
            parent = j.find("parent").get("link")
            child = j.find("child").get("link")
            self._joints.append((name, parent, child, jtype))
            self._parent_to_children_joints[parent].append((child, name))
            self._parent_to_children_links[parent].append(child)
            self._child_links.add(child)
            self._parent_links.add(parent)

    def root_link(self):
        roots = list(self._parent_links - self._child_links)
        if roots:
            return roots[0]
        return self._joints[0][1] if self._joints else None

    def joint_order_by_file(self):
        order = []
        for j in self.root.findall("joint"):
            name = j.get("name")
            jtype = j.get("type")
            if jtype in ("fixed", "floating"):
                continue
            order.append(name)
        return order

    def link_order_by_file(self):
        order = []
        for link in self.root.findall("link"):
            name = link.get("name")
            if name is None:
                continue
            order.append(name)
        return order

    def bfs_joint_order(self):
        root_link = self.root_link()
        if root_link is None:
            return []
        order = []
        q = deque([root_link])
        while q:
            link = q.popleft()
            for child_link, joint_name in self._parent_to_children_joints.get(link, []):
                order.append(joint_name)
                q.append(child_link)
        return order

    def dfs_joint_order(self):
        root_link = self.root_link()
        if root_link is None:
            return []
        order = []
        stack = [root_link]
        while stack:
            link = stack.pop()
            children = self._parent_to_children_joints.get(link, [])
            for child_link, joint_name in reversed(children):
                order.append(joint_name)
                stack.append(child_link)
        return order

    def bfs_link_order(self):
        root_link = self.root_link()
        if root_link is None:
            return []
        order = []
        q = deque([root_link])
        while q:
            link = q.popleft()
            order.append(link)
            for child in self._parent_to_children_links.get(link, []):
                q.append(child)
        return order


if __name__ == "__main__":
    urdf_path = "deploy_mujoco/assets/Q1/urdf/Q1_wo_hand_rl.urdf"
    urdf = UrdfGraph(urdf_path)

    isaac_sim_joint_names = urdf.bfs_joint_order()
    isaac_sim_link_names = urdf.bfs_link_order()
    mujoco_joint_names = urdf.joint_order_by_file()
    mujoco_link_names = urdf.link_order_by_file()

    print(isaac_sim_joint_names)
    print(isaac_sim_link_names)
    print(mujoco_joint_names)
    print(mujoco_link_names)
