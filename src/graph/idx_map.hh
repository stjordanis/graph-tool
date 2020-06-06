// graph-tool -- a general graph modification and manipulation thingy
//
// Copyright (C) 2006-2020 Tiago de Paula Peixoto <tiago@skewed.de>
//
// This program is free software; you can redistribute it and/or modify it under
// the terms of the GNU Lesser General Public License as published by the Free
// Software Foundation; either version 3 of the License, or (at your option) any
// later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
// details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

#ifndef IDX_MAP_HH
#define IDX_MAP_HH

#include <vector>
#include <utility>
#include <limits>

template <class Key, class T>
class idx_map
{
public:
    typedef Key key_type;
    typedef T mapped_type;
    typedef std::pair<const Key, T> value_type;
    typedef typename std::vector<std::pair<Key,T>>::iterator iterator;
    typedef typename std::vector<std::pair<Key,T>>::const_iterator const_iterator;

    template <class P>
    std::pair<iterator,bool> insert(P&& value)
    {
        if (_pos.size() <= size_t(value.first))
            _pos.resize(value.first + 1, _null);
        size_t& idx = _pos[value.first];
        if (idx == _null)
        {
            idx = _items.size();
            _items.push_back(value);
            return std::make_pair(begin() + idx, true);
        }
        else
        {
            _items[idx].second = value.second;
            return std::make_pair(begin() + idx, false);
        }
    }

    size_t erase(const Key& k)
    {
        size_t& idx = _pos[k];
        if (idx == _null)
            return 0;
        _pos[_items.back().first] = idx;
        std::swap(_items[idx], _items.back());
        _items.pop_back();
        idx = _null;
        return 1;
    }

    iterator erase(const_iterator pos)
    {
        size_t idx = pos - begin();
        erase(pos->first);
        return begin() + idx;
    }

    T& operator[](const Key& key)
    {
        auto iter = find(key);
        if (iter == end())
            iter = insert(std::make_pair(key, T())).first;
        return iter->second;
    }

    iterator find(const Key& key)
    {
        if (size_t(key) >= _pos.size())
            return end();
        size_t idx = _pos[key];
        if (idx == _null)
            return end();
        return begin() + idx;
    }

    const iterator find(const Key& key) const
    {
        if (size_t(key) >= _pos.size())
            return end();
        size_t idx = _pos[key];
        if (idx == _null)
            return end();
        return begin() + idx;
    }

    void clear()
    {
        for (auto k : _items)
            _pos[k.first] = _null;
        _items.clear();
    }

    void shrink_to_fit()
    {
        _items.shrink_to_fit();
        if (_items.empty())
            _pos.clear();
        _pos.shrink_to_fit();
    }

    iterator begin() { return _items.begin(); }
    iterator end() { return _items.end(); }
    const_iterator begin() const { return _items.begin(); }
    const_iterator end() const { return _items.end(); }

    size_t size() { return _items.size(); }
    bool empty() { return _items.empty(); }

private:
    std::vector<std::pair<Key,T>> _items;
    std::vector<size_t> _pos;
    static constexpr size_t _null = std::numeric_limits<size_t>::max();
};

template <class Key, class T>
constexpr size_t idx_map<Key, T>::_null;

template <class Key>
class idx_set
{
public:
    typedef Key key_type;
    typedef Key value_type;
    typedef typename std::vector<Key>::iterator iterator;
    typedef typename std::vector<Key>::const_iterator const_iterator;

    std::pair<iterator,bool> insert(const Key& k)
    {
        if (_pos.size() <= size_t(k))
            _pos.resize(k + 1, _null);
        size_t& idx = _pos[k];
        if (idx == _null)
        {
            idx = _items.size();
            _items.push_back(k);
            return std::make_pair(begin() + idx, true);
        }
        else
        {
            return std::make_pair(begin() + idx, false);
        }
    }

    size_t erase(const Key& k)
    {
        size_t& idx = _pos[k];
        if (idx == _null)
            return 0;
        _pos[_items.back()] = idx;
        std::swap(_items[idx], _items.back());
        _items.pop_back();
        idx = _null;
        return 1;
    }

    iterator erase(const_iterator pos)
    {
        size_t idx = pos - begin();
        erase(pos->first);
        return begin() + idx;
    }

    iterator find(const Key& key)
    {
        if (size_t(key) >= _pos.size())
            return end();
        size_t idx = _pos[key];
        if (idx == _null)
            return end();
        return begin() + idx;
    }

    const iterator find(const Key& key) const
    {
        if (size_t(key) >= _pos.size())
            return end();
        size_t idx = _pos[key];
        if (idx == _null)
            return end();
        return begin() + idx;
    }

    void clear()
    {
        for (auto k : _items)
            _pos[k] = _null;
        _items.clear();
    }

    void shrink_to_fit()
    {
        _items.shrink_to_fit();
        if (_items.empty())
            _pos.clear();
        _pos.shrink_to_fit();
    }

    iterator begin() { return _items.begin(); }
    iterator end() { return _items.end(); }
    const_iterator begin() const { return _items.begin(); }
    const_iterator end() const { return _items.end(); }

    size_t size() { return _items.size(); }
    bool empty() { return _items.empty(); }

private:
    std::vector<Key> _items;
    std::vector<size_t> _pos;
    static constexpr size_t _null = std::numeric_limits<size_t>::max();
};

template <class Key>
constexpr size_t idx_set<Key>::_null;

#endif // IDX_MAP_HH
