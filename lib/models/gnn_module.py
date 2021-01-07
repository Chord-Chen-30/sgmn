# -*- coding: utf-8 -*-
import sys, os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init

'''
    不管是哪种GNN模块，我们的输入都只有:
    --feature: (64, 57, 2048)整个batch里每个image里每个bound box(node)的feature.
    --cxt_feats: (64, 57, 5, 2048)多出来的一维对应上面每个bound box的5个neighbor的feature.
'''
'''
    我可以把输入的feature:(64,57,2048)转成(64*57, 2048)这样一个二维矩阵，表示67*57个node，一个node的特征维度是2048
    同样，把输入的cxt_feats:(64, 57, 5, 2048)转成(67*57,5,2048)，第一个维度表示节点的数量，第二个维度表示邻居节点的数量，第三个表示输入特征的维度
    所以要再传入前执行如下操作：
    feat_input = feature.view(-1, feature.shape[2])     # the new shape is (64*57, 2048)
    cxt_feats_input = cxt_feats.view(-1, cxt_feats.shape[2], cxt_feats.shape[3]) # the new shape is (64*57, 5, 2048)
'''
#-----Below is GraphSage-----#
# 聚合：
class NeighborAggregator(nn.Module):
    '''
    聚合节点邻居
    Args:
        input_dim: 输入特征的维度， 2048
        output_dim: 输出特征的维度， 2048 (keep the same)
        use_bias: 是否使用偏置(default: {False})
		aggr_method: 邻居聚合方式(default:{mean})
    '''
    def __init__(self, input_dim, output_dim, use_bias=False, aggr_method="mean"):
        super(NeighborAggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim  #现在这个output_dim还需要再想一下到底是什么
        self.use_bias = use_bias
        self.aggr_method = aggr_method
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_dim))
        self.reset_parameters() #re-initialize the parameters, see below

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, cxt_feats):
        # Attention: we have reshaped cxt_feats, so it is (bs*n, 5, dim) now.
        # cxt_feats is the features of neighbors of each node.
        # we only use "mean" aggr method in this module.
        if self.aggr_method == "mean":
            aggr_cxt_feats = cxt_feats.mean(dim=1) # Now, its shape is (bs*n, dim)
        elif self.aggr_method == "sum":
            aggr_cxt_feats = cxt_feats.sum(dim=1)
        elif self.aggr_method == "max":
            aggr_cxt_feats = cxt_feats.max(dim=1)

        neighbor_hidden = torch.matmul(aggr_cxt_feats, self.weight)
        
        if self.use_bias:
            neighbor_hidden += self.bias
        
        return neighbor_hidden
        
    def extra_repr(self):
        return 'in_features={}, out_features={}, aggr_method={}'.format(
            self.input_dim, self.output_dim, self.aggr_method)

class GraphSage(nn.Module):
    """
        GraphSage层定义
        Args:
            input_dim: 输入特征的维度
            hidden_dim: 隐层特征的维度，
                当aggr_hidden_method=sum, 输出维度为hidden_dim
                当aggr_hidden_method=concat, 输出维度为hidden_dim*2
            activation: 激活函数
            aggr_neighbor_method: 邻居特征聚合方法，["mean", "sum", "max"]
            aggr_hidden_method: 节点特征的更新方法，["sum", "concat"]
    """
    def __init__(self, input_dim, hidden_dim, activation=F.relu, 
                 aggr_neighbor_method="mean", aggr_hidden_method="sum"):
        super(GraphSage, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.aggr_neighbor_method = aggr_neighbor_method # 聚合邻居特征的方法
        self.aggr_hidden_method = aggr_hidden_method # 更新节点自身特征的方法
        self.activation = activation
        self.aggregator = NeighborAggregator(input_dim, hidden_dim, aggr_method=aggr_neighbor_method) #初始化一个聚合邻居节点特征的模块，在上面定义好了
        self.weight = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
    
    def forward(self, src_node_features, neighbor_node_features):
        '''
        src_node_features: (num_node, dim)= (64*57, 2048)
        neighbor_node_features: (num_node, num_neighbor, dim)=(64*57, 5, 2048)
        '''
        #首先，邻居节点的隐藏层neighbor_hidden由定义的聚合器self.aggregator计算得到,该部分侧重于节点的环境信息（图关联信息）。
        neighbor_hidden  = self.aggregator(neighbor_node_features) #输出维度是: (num_nodes, hidden_dim)=(64*57, 2048)
        #self_hidden计算的是节点的自身属性。
        self_hiddden = torch.matmul(src_node_features, self.weight)#输出维度是: (num_nodes, hidden_dim)=(64*57, 2048)
        #环境信息和自身属性的结合有两种方式，一种是矩阵相加（sum），一种是矩阵相接（concat）。
        if self.aggr_hidden_method == "sum":
            hidden = self_hiddden*0.8 + neighbor_hidden*0.2
        elif self.aggr_hidden_method == "concat":
            hidden = torch.cat([self_hiddden, neighbor_hidden], dim=1)

        #最后通过设置的激活函数activation计算得到结果
        if self.activation:
            return self.activation(hidden) # relu will not change the shape, so the shape we return is (num_nodes, dim)=(64*57, 2048)
        else:
            return hidden
        #在外面记得把维度转回来：(64*57, -1)->(64, 57, -1)
    
    def extra_repr(self):
        output_dim = self.hidden_dim if self.aggr_hidden_method == "sum" else self.hidden_dim * 2
        return 'in_features={}, out_features={}, aggr_hidden_method={}'.format(
            self.input_dim, output_dim, self.aggr_hidden_method)

#OK, we can shut down here, we have got what we want.
#-----Above is GraphSage-----#

#-----Below is GAT-----#
'''
https://blog.csdn.net/qq_26593695/article/details/109538241
定义节点vi的特征表示为hi，hi属于R^d, d为特征维度， 聚合后的新节点为hi'
邻居节点vj聚合到vi的权重系数eij为: a(Whi, Whj)
输入是节点集合h={h1, h2, ..., hN}   hi属于R^F
输出是节点集合h'={h1',h2',...,hN'}  hi'属于R^F'
W属于R^(F*F')是该层节点特征变换的权重参数. a(·)是计算两个节点相关度的函数，比如使用内积作为相关度，我们选择一个单层的全连接层，||表示拼接操作。
注意，对于节点vi来说，vi也是自己的邻居
eij=LeakyReLU(aT[Whi||Whj])   #所以在我们的project中，j就是1到5，因为每个node只有5个邻居节点。
为了使不同节点间的权重系数易于比较，我们使用softmax函数对所有计算出的权重进行归一化:
aij=Softmax_i(eij)= exp(eij) / sum(forall neighbor k: exp(eik))
aij就是我们的所求，即聚合的权重系数，softmax函数保证了所有系数的和为1.
计算完上述归一化注意力系数后，再计算与之对应的特征的线性组合，作为每个节点的最终输出特征。
应用非线性激活函数：
hi'=sigma(sum_(vi的所有邻居vj)(aij Whj))

'''
'''
在外面调用的方式:
model = GraphAttentionLayer()
model(feature, )
'''
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features      # it is 2048 in our project
        self.out_features = out_features    # it is 2048 in our project
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))   # (2048, 2048)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))           # (2048*2, 1)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, feature, cxt_idx, offset_idx, cxt_idx_mask, bs, n):
        '''
        Args:
            --src_node_features: 节点特征矩阵 (num_node, dim)= (64*57, 2048)
            --neighbor_node_features: 邻居特征矩阵 (num_node, num_neighbor, dim)=(64*57, 5, 2048)
        '''
        hidden_feature = torch.mm(feature, self.W)  #shape:(64*57, 2048)
        hidden_feature_ori_shape = hidden_feature.view(bs, n, -1) # hidden feature with original shapeL (64, 57, 2048), used for select cxt_feats
        # select neighor node features:
        cxt_feats = torch.index_select(hidden_feature_ori_shape.view(bs*n, -1), 0, (offset_idx+cxt_idx).view(-1))
        cxt_feats = cxt_feats.view(bs, n, 5, -1) * cxt_idx_mask.unsqueeze(3).float() #shape:(64, 57, 5, 2048)  

        cxt_feats = cxt_feats.view(bs*n, -1) # shape:(64*57, 5*2048) 
        node_and_neighbor =  torch.cat([hidden_feature, cxt_feats], dim=1) # shape: (64*57, 6*2048) 因为对于节点vi来说，vi也是自己的邻居，所以之后也要算vi对vi的attention
        node_and_neighbor = node_and_neighbor.view(bs*n*6, 2048)  # shape: (64*57*6, 2048)

        # Now, concat hidden_feature and node_and_neighbor. Before that, we need to repeat hidden_feature 6 times, 6 is the number of neighbors(including itself).
        repeat_hidden_feature = hidden_feature.repeat(1, 6).view(-1,2048) #Now, its shape is: (64*57*6, 2048)
        concat = torch.cat([repeat_hidden_feature, node_and_neighbor], dim=1).view(bs*n, 6, -1) # shape: (64*57*6, 2048*2)->(64*57, 6, 2048*2)
        # 通过刚刚的拼接矩阵与权重矩阵a相乘计算每两个样本之间的相关性权重，最后再根据邻接矩阵置零没有连接的权重
        e = self.leakyrelu(torch.matmul(concat, self.a).squeeze(2)) # (64*57, 6, 1)->(64*57, 6) 那么现在每一行是一的node对6个邻居权重值
        attention = F.softmax(e, dim=1)  # (64*57, 6)
        attention = F.dropout(attention, self.dropout, training=self.training) # (64*57,6) #现在每一行是一个node对6个邻居的attention值
        node_and_neighbor = node_and_neighbor.view(bs*n, 6, 2048)
        h_prime = torch.einsum('ij,ijk->ik', (attention, node_and_neighbor))  #(64*57, 6)*(64*57, 6, 2048) should be (64*57, 2048)
        # h_prime = torch.matmul(attention, hidden_feature) #(64*57, 2048)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.6, alpha=0.2):
        """
        Dense version of GAT.
        :param nfeat: 输入特征的维度
        :param nhid:  输出特征的维度
        :param nclass: 分类个数
        :param dropout: dropout
        :param alpha: LeakyRelu中的参数
        :param nheads: 多头注意力机制的个数
        """
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
        self.add_module('attention_{}'.format(0), self.attentions)
        # self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False) #we do not need this layer in this project.
    
    def forward(self, x, cxt_idx, offset_idx, cxt_idx_mask, bs, n):
        x = F.dropout(x, self.dropout, training=self.training) 
        x = self.attentions(x, cxt_idx, offset_idx, cxt_idx_mask, bs, n) #(2708,64)
        x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)   #(64*57, 2048)

#-----Above is GAT-----#