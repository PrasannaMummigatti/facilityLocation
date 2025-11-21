import torch
import torch.nn.functional as F
from torch_geometric.nn import NNConv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def getImage(path, zoom=1):
    return OffsetImage(plt.imread(path), zoom=0.08)

# ============================================================
# 1. MODEL (must match training version)
# ============================================================
class FLPGNN_EdgeAttr(torch.nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, edge_attr_dim=1):
        super().__init__()
        self.edge_mlp1 = torch.nn.Sequential(
            torch.nn.Linear(edge_attr_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, input_dim * hidden_dim)
        )
        self.conv1 = NNConv(
            in_channels=input_dim,
            out_channels=hidden_dim,
            nn=self.edge_mlp1,
            aggr='mean'
        )

        self.edge_mlp2 = torch.nn.Sequential(
            torch.nn.Linear(edge_attr_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, hidden_dim * hidden_dim)
        )
        self.conv2 = NNConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            nn=self.edge_mlp2,
            aggr='mean'
        )

        self.lin = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_attr):
        h = F.relu(self.conv1(x, edge_index, edge_attr))
        h = F.relu(self.conv2(h, edge_index, edge_attr))
        out = self.lin(h)
        return out.squeeze(-1)


# ============================================================
# 2. Build Graph Tensors
# ============================================================
def build_graph_tensors(customer_coords, facility_coords, customer_demand):
    M = customer_coords.shape[0]
    F = facility_coords.shape[0]

    # Node features: [x, y, demand]
    cust_feats = np.hstack([customer_coords, customer_demand.reshape(-1, 1)])
    fac_feats = np.hstack([facility_coords, np.zeros((F, 1))])
    node_features = np.vstack([cust_feats, fac_feats]).astype(np.float32)
    x = torch.tensor(node_features, dtype=torch.float32)

    # Bipartite edges + distances
    edge_list = []
    edge_attr_list = []

    for i in range(M):
        for j in range(F):
            d = np.linalg.norm(customer_coords[i] - facility_coords[j])
            edge_list.append([i, M + j])
            edge_attr_list.append([d])
            edge_list.append([M + j, i])
            edge_attr_list.append([d])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)

    return x, edge_index, edge_attr, M, F


# ============================================================
# 3. Assign Customers → Chosen Facilities
# ============================================================
def assign_customers(customer_coords, facility_coords, chosen_idx):
    chosen_idx = np.array(chosen_idx, dtype=int)
    M = customer_coords.shape[0]
    assigned = np.empty(M, dtype=int)

    for i in range(M):
        dists = np.linalg.norm(customer_coords[i] - facility_coords[chosen_idx], axis=1)
        nearest = chosen_idx[np.argmin(dists)]
        assigned[i] = nearest

    return assigned


# ============================================================
# 4. EXAMPLE DATA (replace with your real data)
# ============================================================
customer_coords = np.array([
    [1,2],[3,3],[5,1],[4,3],[5,7],[5,2],[4,3],[5,8],[4,5]
], float)

facility_coords = np.array([
    [1,8],[3,5],[2,5],[5,6]
], float)

customer_demand = np.array([20,4,30,10,40,50,10,300,20], float)

p = 2  # number of facilities to open


# ============================================================
# 5. Build tensors
# ============================================================
x, edge_index, edge_attr, M, num_facilities  = build_graph_tensors(
    customer_coords, facility_coords, customer_demand
)

# ============================================================
# 6. Load trained model
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FLPGNN_EdgeAttr(input_dim=3, hidden_dim=64, edge_attr_dim=1).to(device)
model.load_state_dict(torch.load("./GNN/flp_gnn_model_edgeattr.pt", map_location=device))
model.eval()

with torch.no_grad():
    x = x.to(device)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    out = model(x, edge_index, edge_attr).cpu().numpy()

# Extract facility node predictions
facility_pred = out[M : M + num_facilities] # out[M:]  # facility nodes only, shape [F]
top_p_idx = np.argsort(facility_pred)[-p:]

print("Open facilities →", top_p_idx)
print("Scores →", facility_pred[top_p_idx])


# ============================================================
# 7. Assign customers to selected facilities
# ============================================================
assigned_fac = assign_customers(customer_coords, facility_coords, top_p_idx)


# ============================================================
# 8. Visualization
# ============================================================
plt.figure(figsize=(8,6), facecolor='lightyellow')

# Customers
plt.scatter(customer_coords[:,0], customer_coords[:,1],
            s=customer_demand*3, c="blue", label="Customers")

# All facilities
plt.scatter(facility_coords[:,0], facility_coords[:,1],
            s=80, c="gray", marker="s", label="All Facilities")

for i, (x, y) in enumerate(zip(facility_coords[:,0], facility_coords[:,1])):
    ab = AnnotationBbox(getImage('house-Gray.png'), (x, y), frameon=False)
    plt.gca().add_artist(ab)


# Open facilities
plt.scatter(facility_coords[top_p_idx,0], facility_coords[top_p_idx,1],
            s=80, c="red", marker="s", label="Open Facilities")
for i, (x, y) in enumerate(zip(facility_coords[top_p_idx,0], facility_coords[top_p_idx,1])):
    ab = AnnotationBbox(getImage('house-xxl.png'), (x, y), frameon=False)
    plt.gca().add_artist(ab)



# Draw assignment lines
for i in range(M):
    f = assigned_fac[i]
    plt.plot(
        [customer_coords[i,0], facility_coords[f,0]],
        [customer_coords[i,1], facility_coords[f,1]],
        "k--", alpha=0.5
    )

plt.title(f"GNN Prediction with Distance Edge Features (p={p})")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(facecolor='lightyellow', edgecolor='none', markerscale=0.4,loc='lower center')
plt.grid(False)
plt.axis('off')
plt.show()
