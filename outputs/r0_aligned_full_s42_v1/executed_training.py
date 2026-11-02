def test_edge(loader, model, device='cuda'):
    model.eval()
    correct = 0
    num_graphs = 0
    for batch in loader:
        batch.to(device)
        with torch.no_grad():
            pred = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict, batch).max(dim=1)[1]
            label = batch.y
        correct += pred.eq(label).sum().item()
        num_graphs += batch.num_graphs
    return correct / num_graphs

def train_with_edge_Att(train_loader, model, args, device='cuda'):
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    best_model = None
    max_val = -1
    for epoch in range(args['epochs']):
        total_loss = 0
        model.train()
        num_graphs = 0
        for batch in tqdm(train_loader):
            batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict, batch)
            label = batch.y
            loss = model.loss(pred, label)
            loss.backward()
            optimizer.step()
            _step_monitor(model, batch, optimizer)
            total_loss += loss.item() * batch.num_graphs
            num_graphs += batch.num_graphs
        total_loss /= num_graphs
        train_acc = test_edge(train_loader, model, device)
        current_lr = optimizer.param_groups[0]['lr']
        log = 'Epoch {}: Train: {:.4f}, Loss: {:.4f}, Lr: {:.6f}'
        print(log.format(epoch + 1, train_acc, total_loss, current_lr))
        _epoch_monitor(epoch + 1, model, optimizer, total_loss, train_acc, num_graphs)
