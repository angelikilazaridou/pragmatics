local cjson = require 'cjson'
local utils = {}

-- Assume required if default_value is nil
function utils.getopt(opt, key, default_value)
	if default_value == nil and (opt == nil or opt[key] == nil) then
    		error('error: required key ' .. key .. ' was not provided in an opt.')
  	end
  	if opt == nil then return default_value end
  	local v = opt[key]
  	if v == nil then v = default_value end
  	return v
end

function utils.read_json(path)
  	local file = io.open(path, 'r')
  	local text = file:read()
  	file:close()
  	local info = cjson.decode(text)
  	return info
end

function utils.write_json(path, j)
  	-- API reference http://www.kyne.com.au/~mark/software/lua-cjson-manual.html#encode
  	cjson.encode_sparse_array(true, 2, 10)
  	local text = cjson.encode(j)
  	local file = io.open(path, 'w')
  	file:write(text)
  	file:close()
end


return utils
