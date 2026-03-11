import pandapower
from pandapower.networks import case14
from pandapower_env.substation.multi_bb_substaion import create_nbb

net = case14()
print(net.bus)
create_nbb(net, 2, 3)
print(net.bus)
