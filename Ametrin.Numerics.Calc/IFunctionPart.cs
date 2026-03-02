using System.Linq;

namespace Ametrin.Numerics.Calc;

public interface IFunctionPart
{
    public IndexDescriptor Size { get; }

    // public void Backprop(IFunctionPart resultGradient)
}

public interface IScalarPart : IFunctionPart
{
    IndexDescriptor IFunctionPart.Size => IndexDescriptor.Zero;
}

internal sealed record AddPart(IFunctionPart Left, IFunctionPart Right) : IFunctionPart
{
    public IndexDescriptor Size => Left.Size;
}

internal sealed record TensorPart(Identifier Name, IndexDescriptor Size) : IFunctionPart;

internal sealed record CellScalarPart(TensorPart Tensor, IndexDescriptor Index) : IScalarPart;
internal sealed record VariableScalarPart(Identifier Name) : IScalarPart;
internal sealed record ConstantScalarPart(decimal Value) : IScalarPart;
public sealed record Identifier(string Name, Context Context);

public sealed record IndexDescriptor(ImmutableArray<IScalarPart> Dimensions)
{
    public int Rank => Dimensions.Length;
    // public IndexDescriptor Resolve() => this;

    public static IndexDescriptor Zero { get; } = new([]);

    public bool Equals(IndexDescriptor? other)
    {
        if (other is null)
        {
            return false;
        }

        // SequenceEqual does a length comparison
        return Dimensions.SequenceEqual(other.Dimensions);
    }

    public override int GetHashCode()
    {
        var hash = new HashCode();
        foreach (var part in Dimensions)
        {
            hash.Add(part);
        }
        return hash.ToHashCode();
    }
}

public static class Factory
{
    public static IFunctionPart Zero => new ConstantScalarPart(0);
    public static IFunctionPart One => new ConstantScalarPart(1);

    public static IFunctionPart Constant(decimal value) => value switch
    {
        0 => Zero,
        1 => One,
        _ => new ConstantScalarPart(value),
    };

    public static IFunctionPart Add(IFunctionPart left, IFunctionPart right)
    {
        if (left.IsZero) return right;
        if (right.IsZero) return left;
        return new AddPart(left, right);
    }

    extension(IFunctionPart part)
    {
        internal bool IsZero => part is ConstantScalarPart { Value: 0 };
    }
}


public sealed class Context(string name, FrozenSet<Context> accessible)
{
    public string Name { get; } = name;
    public FrozenSet<Context> Accessible { get; } = accessible;
    private Dictionary<string, IFunctionPart> Values { get; } = [];

    public IFunctionPart this[Identifier identifier]
    {
        get
        {
            if (identifier.Context == this)
            {
                if (Values.TryGetValue(identifier.Name, out var part))
                {
                    return part;
                }

                throw new InvalidOperationException($"Undefined part {identifier.Name}");
            }

            if (Accessible.Contains(identifier.Context))
            {
                return identifier.Context[identifier];
            }

            throw new InvalidOperationException($"Cannot access {identifier.Context.Name}.{identifier.Name} from {Name}");
        }
    }

    private Identifier CreateIdentifer(string name) => new(name, this);
    public Identifier AssignPart(string name, IFunctionPart part)
    {
        if (Values.ContainsKey(name))
        {
            throw new InvalidOperationException($"{Name}.{name} already defined");
        }

        Values[name] = part;

        return CreateIdentifer(name);
    }

    public IFunctionPart GetReference(Identifier identifier)
    {
        var part = this[identifier];

        if (part is TensorPart or CellScalarPart or ConstantScalarPart or VariableScalarPart)
        {
            return part;
        }

        var size = part.Size;
        return size.Rank switch
        {
            0 => new VariableScalarPart(identifier),
            _ => new TensorPart(identifier, size),
        };
    }

    public bool CanAccess(Identifier identifier)
        => identifier.Context == this || Accessible.Contains(identifier.Context);

    private void ValidateAccess(Identifier identifier)
    {
        if (CanAccess(identifier)) return;
        throw new InvalidOperationException($"Cannot access {identifier.Context.Name}.{identifier.Name} from {Name}");
    }
}

public sealed class LayerDefinition
{
    public Context Layer { get; }
    public Context Snapshot { get; }
    public Context Gradients { get; }
    public Context Forward { get; }
    public Context Backward { get; }

    public LayerDefinition()
    {
        Layer = new("layer", []);
        Snapshot = new("snapshot", [Layer]);
        Gradients = new("gradients", [Layer]);
        Forward = new("forward", [Layer, Snapshot]);
        Backward = new("backward", [Layer, Snapshot, Gradients]);
    }
}