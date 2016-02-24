--local input_dir = "/home/angeliki/git/pragmatics/DATA/"
local input_dir = "/home/thenghiapham/work/project/pragmatics/DATA/"
local mat_file = input_dir .. "visVecs/fc7.mat"
local mat_label_file = input_dir .. "visVecs/fc7labels.txt"
local concept_file = input_dir .. "visAttCarina/raw/concepts.txt"
local concept_vector_file = input_dir .. "visAttCarina/raw/vectors.txt"
local split_file = input_dir .. "visAttCarina/processed/0shot/README.txt"

--local output_dir = "/home/angeliki/git/pragmatics/DATA/visAttCarina/processed/prop_baseline/"
local output_dir = "/home/thenghiapham/work/project/pragmatics/DATA/visAttCarina/processed/prop_baseline/"
local train_file = output_dir .. "train.txt"
local test_file = output_dir .. "test.txt"

local function readlists(input_file)
  local i_stream = io.open(input_file, 'r')
  local outputs = {}
  local line = i_stream:read()
  while (line ~= nil) do
    table.insert(outputs, line)
    line = i_stream:read()
  end 
  i_stream:close()
  return outputs
end

local function array2dict(array)
  local dict = {}
  for i=1,#array do
    dict[array[i]] = i
  end
  return dict
end

local concepts = readlists(concept_file)
local all_concepts = readlists(mat_label_file)
local all_concept_dict = array2dict(all_concepts)
local all_visual_vs = readlists(mat_file)
local concept_visual_vs = {}
for i=1,#concepts do
  local concept = concepts[i]
  local c_index = all_concept_dict[concept]
  local c_visual_v = all_visual_vs[c_index]
  table.insert(concept_visual_vs, c_visual_v)
end

local function read_dict(rm_file)
  local i_stream = io.open(rm_file, 'r')
  i_stream:read()
  i_stream:read()
  i_stream:read()
  local line = i_stream:read()
  local length = string.len(line)
  line = string.sub(line, 2, length - 1)
  local elements = line:split(", ")
  local test_concept = {}
  for i=1,#elements do
    local element = elements[i]
    local es = element:split(": ")
    table.insert(test_concept, es[2])
  end 
  i_stream:close()
  return test_concept
end

local test_concepts = read_dict(split_file)
local test_concept_dict = array2dict(test_concepts)
local concept_properties = readlists(concept_vector_file)

local train_stream = io.open(train_file, "w")
local test_stream = io.open(test_file, "w")
for i=1,#concepts do
  local concept = concepts[i]
  if test_concept_dict[concept] ~= nil then
    test_stream:write(concept_visual_vs[i] .. " " .. concept_properties[i] .. "\n")
  else
    train_stream:write(concept_visual_vs[i] .. " " .. concept_properties[i] .. "\n")
  end
end

train_stream:close()
test_stream:close()