class NIDSDataset(Dataset):
    """A dataset classfor  generates graph data objects from a provided
    CSV File/Files. Build uopn torch_geometric.data.Dataset base class for 
    creating graph datasets.

    Args:
        root (str): Root directory where the dataset should be saved.
        
        label_dict (Dict): Dictionary for assigning labels to each attack class.

        filename(List[str]): List of CSV file paths to be used for the development of 
            graph objects.

        skip_processing (bool): If set to `True`, skips the generation of graph 
            objects and utilizes the ones present in the root directory. 
            (default: `False`)

        test (bool): If set to `True`, generates data objects for testing by 
            creating data objects with a test suffix. 
            (default: `False`)

        index_num (int): Index to start with saving the grapgh object. Useful 
            when creating continuing grapgh object and you have to run the same
            class twice.
            (default: 0)

        include_packetflag (bool): If set to True, will also include Flags status
            as packet node features.
            (default: False)

        include_packetpayload (bool): If set to False, will not include payload 
            information as a packet node features.
            (default: True) 

        single_file (bool): If set to True, the provided CSV files is a single 
            file with Label column within CSV.
            (default: False)      

        transform (callable, optional): A function/transform that takes in a
            :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` object and returns a
            transformed version.
            The data object will be transformed before every access.
            (default: :obj:`None`)

        pre_transform (callable, optional): A function/transform that takes in
            a :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` object and returns a
            transformed version.
            The data object will be transformed before being saved to disk.
            (default: :obj:`None`)
    """

    def __init__(self, root, label_dict, filename, index_num=0, skip_processing=False, include_packetflag=False, include_packetpayload=True, test=False, single_file=False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.test = test
        self.filename = filename
        self.include_packetflag = include_packetflag
        self.include_packetpayload = include_packetpayload
        self.index = index_num
        self.label_dict = label_dict
        self.skip_processing = skip_processing
        self.length = 0
        self.single_file = single_file
        super(NIDSDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        if self.skip_processing:
            if self.test:
                list_process = glob.glob(os.path.join(self.root, 'processed/data_test*'))
                self.length = len(list_process)
                return list_process[0]
            else:
                list_process = glob.glob(os.path.join(self.root, 'processed/data*'))
                self.length = len(list_process) - len(glob.glob(os.path.join(self.root, 'processed/data_test*')))
                return list_process[0]
        else:
            return []

    def download(self):
        pass

    def process(self):
        for files in self.raw_paths:
            self.data = pd.read_csv(files)
            print('Reading File ---> ' + os.path.basename(files), file=sys.stderr)
            if not self.single_file:
                self.data.drop(['src_ip', 'src_port', 'dst_ip', 'dst_port', 'ip_version'], axis=1, inplace=True)
                self.data.drop(['bidirectional_bytes', 'bidirectional_first_seen_ms', 'bidirectional_last_seen_ms', 'bidirectional_duration_ms', 'bidirectional_packets', 'src2dst_first_seen_ms', 'src2dst_last_seen_ms', 'dst2src_first_seen_ms', 'dst2src_last_seen_ms', 'id', 'src_mac', 'src_oui', 'dst_mac', 'dst_oui', 'vlan_id', 'tunnel_id', 'bidirectional_syn_packets', 'bidirectional_cwr_packets', 'bidirectional_ece_packets', 'bidirectional_urg_packets', 'bidirectional_ack_packets', 'bidirectional_psh_packets', 'bidirectional_rst_packets', 'bidirectional_fin_packets'], axis=1, inplace=True)
                self.data['expiration_id'] = pd.Categorical(self.data['expiration_id'], categories=[0, -1])
                self.data['protocol'] = pd.Categorical(self.data['protocol'], categories=[1, 2, 6, 17, 58])
                self.data = pd.get_dummies(self.data, prefix=['Exp', 'proto'], columns=['expiration_id', 'protocol'], dtype=int)
                label = self._get_labels(files)
            self.data['udps.payload_data'] = self.data['udps.payload_data'].map(lambda x: x.strip('][').replace("'", '').split(', '))
            self.data['udps.packet_direction'] = self.data['udps.packet_direction'].map(lambda x: x.strip('][').replace("'", '').split(', '))
            self.data['udps.ip_size'] = self.data['udps.ip_size'].map(lambda x: x.strip('][').replace("'", '').split(', '))
            self.data['udps.transport_size'] = self.data['udps.transport_size'].map(lambda x: x.strip('][').replace("'", '').split(', '))
            self.data['udps.payload_size'] = self.data['udps.payload_size'].map(lambda x: x.strip('][').replace("'", '').split(', '))
            self.data['udps.delta_time'] = self.data['udps.delta_time'].map(lambda x: x.strip('][').replace("'", '').split(', '))
            if self.include_packetflag == True:
                self.data['udps.syn'] = self.data['udps.syn'].map(lambda x: x.strip('][').replace("'", '').split(', '))
                self.data['udps.cwr'] = self.data['udps.cwr'].map(lambda x: x.strip('][').replace("'", '').split(', '))
                self.data['udps.ece'] = self.data['udps.ece'].map(lambda x: x.strip('][').replace("'", '').split(', '))
                self.data['udps.urg'] = self.data['udps.urg'].map(lambda x: x.strip('][').replace("'", '').split(', '))
                self.data['udps.ack'] = self.data['udps.ack'].map(lambda x: x.strip('][').replace("'", '').split(', '))
                self.data['udps.psh'] = self.data['udps.psh'].map(lambda x: x.strip('][').replace("'", '').split(', '))
                self.data['udps.rst'] = self.data['udps.rst'].map(lambda x: x.strip('][').replace("'", '').split(', '))
                self.data['udps.fin'] = self.data['udps.fin'].map(lambda x: x.strip('][').replace("'", '').split(', '))
            for (index, flow) in tqdm(self.data.iterrows(), total=self.data.shape[0]):
                flow_node_feats = self._get_flow_node_features(flow)
                packet_node_feats = self._get_packet_node_features(flow)
                contain_edge_index = self._get_contain_edge_index(len(flow['udps.payload_data']))
                link_edge_index = self._get_link_edge_index(len(flow['udps.payload_data']))
                contain_edge_feats = self._get_contain_edge_features(flow)
                link_edge_feats = self._get_link_edge_features(flow)
                if self.single_file:
                    label = torch.tensor(np.asarray(flow['Label']), dtype=torch.int64)
                data = HeteroData()
                data['flow'].x = flow_node_feats
                data['packet'].x = packet_node_feats
                data['flow', 'contain', 'packet'].edge_index = contain_edge_index
                data['packet', 'link', 'packet'].edge_index = link_edge_index
                data['flow', 'contain', 'packet'].edge_attr = contain_edge_feats
                data['packet', 'link', 'packet'].edge_attr = link_edge_feats
                data.y = label
                data = T.ToUndirected()(data)
                if self.test:
                    torch.save(data, os.path.join(self.processed_dir, f'data_test_{self.index}.pt'))
                else:
                    torch.save(data, os.path.join(self.processed_dir, f'data_{self.index}.pt'))
                self.index += 1
            if not self.skip_processing:
                if self.test:
                    list_process = glob.glob(os.path.join(self.root, 'processed/data_test*'))
                    self.length = len(list_process)
                else:
                    list_process = glob.glob(os.path.join(self.root, 'processed/data*'))
                    self.length = len(list_process) - len(glob.glob(os.path.join(self.root, 'processed/data_test*')))

    def _get_flow_node_features(self, flow):
        """ Extract the features for Flow_node
        This will return a 2d array of the shape
        [1 , Node Feature size] """
        if self.single_file:
            flow_data = flow.drop(['Label', 'udps.payload_data', 'udps.delta_time', 'udps.packet_direction', 'udps.ip_size', 'udps.transport_size', 'udps.payload_size', 'udps.syn', 'udps.cwr', 'udps.ece', 'udps.urg', 'udps.ack', 'udps.psh', 'udps.rst', 'udps.fin'])
        else:
            flow_data = flow.drop(['udps.payload_data', 'udps.delta_time', 'udps.packet_direction', 'udps.ip_size', 'udps.transport_size', 'udps.payload_size', 'udps.syn', 'udps.cwr', 'udps.ece', 'udps.urg', 'udps.ack', 'udps.psh', 'udps.rst', 'udps.fin'])
        all_flow_node_feats = np.asarray(flow_data, dtype=float)
        all_flow_node_feats = np.reshape(all_flow_node_feats, (-1, all_flow_node_feats.shape[0]))
        all_flow_node_feats = torch.tensor(all_flow_node_feats, dtype=torch.float32)
        return all_flow_node_feats

    def _get_packet_node_features(self, flow):
        """ Extract the features for packet_node
        Features for each packet is its payload data. We can also incorporate flags data for each packet. 
        This will return a 2d array of the shape
        [Num_packets_in_flow, Node Feature size] """
        dims = 1500
        all_packet_node_feats = []
        for index_value in range(len(flow['udps.payload_data'])):
            packet_feats_combined = []
            if self.include_packetflag == True:
                packet_feats_combined.append(flow['udps.syn'][index_value])
                packet_feats_combined.append(flow['udps.cwr'][index_value])
                packet_feats_combined.append(flow['udps.ece'][index_value])
                packet_feats_combined.append(flow['udps.urg'][index_value])
                packet_feats_combined.append(flow['udps.ack'][index_value])
                packet_feats_combined.append(flow['udps.psh'][index_value])
                packet_feats_combined.append(flow['udps.rst'][index_value])
                packet_feats_combined.append(flow['udps.fin'][index_value])
            if self.include_packetpayload == True:
                byte_array = bytes.fromhex(flow['udps.payload_data'][index_value])
                byte_lst = list(byte_array)
                if len(byte_lst) < dims:
                    packet_feat = np.pad(byte_lst, (0, dims - len(byte_lst)), 'constant')
                else:
                    packet_feat = np.array(byte_lst[0:dims].copy())
                packet_feat = np.abs(np.uint8(packet_feat))
                packet_feats_combined.extend(packet_feat.tolist())
            all_packet_node_feats.append(packet_feats_combined)
        all_packet_node_feats = np.asarray(all_packet_node_feats, dtype=int)
        all_packet_node_feats = torch.tensor(all_packet_node_feats, dtype=torch.float32)
        return all_packet_node_feats

    def _get_contain_edge_features(self, flow):
        """ Extract the features for contain_edge.
        Features for each contain_edge is packet_direction, ip_size, transport_size, payload_size of that particular packet.        
        This will return a 2d array of the shape
        [Number of contain_edges, contain_edge Feature size]
        """
        contain_edge_all_feats = []
        for index_value in range(len(flow['udps.packet_direction'])):
            contain_edge_feat = []
            contain_edge_feat.append(flow['udps.packet_direction'][index_value])
            contain_edge_feat.append(flow['udps.ip_size'][index_value])
            contain_edge_feat.append(flow['udps.transport_size'][index_value])
            contain_edge_feat.append(flow['udps.payload_size'][index_value])
            contain_edge_all_feats.append(contain_edge_feat)
        contain_edge_all_feats = np.asarray(contain_edge_all_feats, dtype=int)
        contain_edge_all_feats = torch.tensor(contain_edge_all_feats, dtype=torch.float32)
        return contain_edge_all_feats

    def _get_link_edge_features(self, flow):
        """ Extract the features for link_edge.
        Features for each link_edge is delta_time between the two packets.
        This will return a 2d array of the shape
        [Number of link_edges, link_edge Feature size]
        """
        link_edge_feats = np.asarray(flow['udps.delta_time'][1:], dtype=int)
        link_edge_feats = torch.tensor(link_edge_feats, dtype=torch.float32)
        return link_edge_feats

    def _get_contain_edge_index(self, length):
        """ Extract the adjacency matrix for the edges connection between flow_node and packet_nodes.
        Contain_Edge links the flow_node to each and every packet_node that it contains.
        For Example: If there are 20 packets in a particular flow then there will be 20 Contain_Edge.
                     As each packet is linked to the flow.
        This will return a matrix / 2d array of the shape
        [2, num_contain_edges]
        """
        Flow = np.zeros(length, dtype=int)
        Packet = np.arange(0, length)
        contain_edge = np.vstack((Flow, Packet))
        contain_edge = torch.tensor(contain_edge, dtype=torch.int64)
        return contain_edge

    def _get_link_edge_index(self, length):
        """ Extract the adjacency matrix for the edges connection between packet_nodes.
        Link_Edge links the packet_node to other packet_node available.
        For Example: Each packet is linked with its following packet. If there are 20 packets then 
                     there will be 19 link_edges.
        This will return a matrix / 2d array of the shape
        [2, num_link_edges]
        """
        packet_ini = np.arange(0, length - 1)
        packet_next = np.arange(1, length)
        link_edge = np.vstack((packet_ini, packet_next))
        link_edge = torch.tensor(link_edge, dtype=torch.int64)
        return link_edge

    def _get_labels(self, file_name):
        name = os.path.basename(file_name)
        label = self.label_dict[name.split('-')[0]]
        return torch.tensor(np.asarray([label]), dtype=torch.int64)

    def len(self):
        return self.length

    def get(self, idx):
        """ _ Equivalent to __getitem__ in pytorch"""
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data
