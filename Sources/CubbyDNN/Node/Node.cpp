#include <CubbyDNN/Core/Graph.hpp>
#include <CubbyDNN/Node/Node.hpp>

namespace CubbyDNN::Node
{
Node::Node(Core::Graph* _graph, std::string_view _name)
    : graph(_graph), name(_name), m_isShapeDirty(true), m_gradientDirty(nullptr)
{
    // Do nothing
}

NodeInput* Node::operator[](const std::string& inputName)
{
    const auto index(m_nodeInputMap.find(inputName));
    return index == m_nodeInputMap.cend() ? nullptr : index->second;
}

const NodeType* Node::Type() const
{
    return graph->nodeTypeManager.Type<Node>();
}

std::string_view Node::TypeName()
{
    return "Node";
}

const Core::Shape& Node::Shape() const noexcept
{
    return m_shape;
}

Core::Span<float> Node::Gradient() const noexcept
{
    return m_gradient.Span();
}

Node& Node::EvalShape()
{
    if (!m_isShapeDirty)
    {
        return *this;
    }

    for (const auto& pair : this->m_nodeInputMap)
    {
        pair.second->InputNode()->EvalShape();
    }

    // TODO: Call EvaluateShape()
    m_isShapeDirty = false;

    return *this;
}

Node& Node::EvalGradient(const Node* dy)
{
    if (m_gradientDirty == dy)
    {
        return *this;
    }

    m_gradient.Resize(EvalShape().m_shape.Size());

    if (dy == this)
    {
        m_gradient.Span().FillOne();
        m_gradientDirty = dy;

        return *this;
    }

    m_gradient.Span().FillZero();

    for (const auto* revNodeInput : m_revNodeInputList)
    {
        if (revNodeInput->node == dy || revNodeInput->node->HasRevDeps(dy))
        {
            revNodeInput->backwardOp(dy);
        }
    }

    m_gradientDirty = dy;

    return *this;
}
}  // namespace CubbyDNN::Node