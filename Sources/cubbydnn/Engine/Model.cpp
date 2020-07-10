// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Engine/Model.hpp>

namespace CubbyDNN::Graph
{
Model::Model(NumberSystem numericType)
    : m_numericType(numericType)
{
}

UnitId Model::PlaceHolder(const Shape& shape, const std::string& name,
                          Compute::Device device)
{
    UnitId subjectUnitId{ UnitType(UnitBaseType::Source, name) };
    UnitMetaData unitMetaData(subjectUnitId, {}, {},
                              {}, shape, {}, {},
                              m_numericType,
                              std::move(device));
    m_unitManager.AppendUnit(std::move(unitMetaData));
    return subjectUnitId;
}

UnitId Model::Dense(const UnitId& input, std::size_t units,
                    std::unique_ptr<Initializer> weightInitializer,
                    std::unique_ptr<Initializer> biasInitializer,
                    const std::string& name,
                    Compute::Device device)
{
    UnitId subjectUnitId{ UnitType(UnitBaseType::Hidden, "Dense"), m_id++,
                          name };
    const auto previousOutputShape = m_unitManager.GetUnitOutputShape(input.Id);
    auto weightShape = previousOutputShape;
    weightShape.SetNumRows(units);
    auto biasShape = weightShape;
    biasShape.SetNumCols(1);
    biasShape.SetNumRows(units);

    auto outputShape = previousOutputShape;
    outputShape.SetNumRows(units);

    std::unordered_map<std::string, std::unique_ptr<Initializer>>
        initializerMap;

    initializerMap["weight"] = std::move(weightInitializer);
    initializerMap["bias"] = std::move(biasInitializer);

    UnitMetaData unitMetaData(subjectUnitId,
                              { { "weight", weightShape },
                                { "bias", biasShape } },
                              std::move(initializerMap),
                              { previousOutputShape }, outputShape,
                              { input }, {}, m_numericType,
                              std::move(device));

    m_unitManager.AppendUnit(std::move(unitMetaData));
    return subjectUnitId;
}

UnitId Model::Activation(const UnitId& input, const std::string& activationName,
                         const std::string& name, Compute::Device device)
{
    UnitId subjectUnitId{ UnitType(UnitBaseType::Hidden, name), m_id++, name };
    const auto previousOutputShape = m_unitManager.GetUnitOutputShape(input.Id);

    UnitMetaData unitMetaData(subjectUnitId, {}, {},
                              { previousOutputShape },
                              { previousOutputShape }, { input },
                              ParameterPack({}, {},
                                            { { "activationName", activationName } }),
                              m_numericType,
                              std::move(device));

    m_unitManager.AppendUnit(std::move(unitMetaData));
    return subjectUnitId;
}

void Model::Compile(const std::string& optimizer, ParameterPack optimizerParams)
{
    m_unitManager.Compile(optimizer, optimizerParams);
}
} // namespace CubbyDNN